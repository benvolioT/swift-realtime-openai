import Foundation
@preconcurrency import AVFoundation

public enum ConversationError: Error {
	case sessionNotFound
	case converterInitializationFailed
}

/// Protocol defining methods to handle server events.
public protocol ConversationDelegate: AnyObject {
    func conversationDidUpdateIsListening(_ conversation: Conversation, isListening: Bool)
    func conversationDidUpdateIsPlaying(_ conversation: Conversation, isPlaying: Bool)
    func conversationDidUpdateIsUserSpeaking(_ conversation: Conversation, isUserSpeaking: Bool)
    func conversationDidReceiveEvent(_ conversation: Conversation, event: ServerEvent)
}

@Observable
public final class Conversation: @unchecked Sendable {
    @ObservationIgnored private var delegates: NSHashTable<AnyObject> = NSHashTable.weakObjects()
    
    private let client: RealtimeAPI
	@MainActor private var cancelTask: (() -> Void)?
	private let errorStream: AsyncStream<ServerError>.Continuation

	private let audioEngine = AVAudioEngine()
	private let playerNode = AVAudioPlayerNode()
	private let queuedSamples = UnsafeMutableArray<String>()
	private let apiConverter = UnsafeInteriorMutable<AVAudioConverter>()
	private let userConverter = UnsafeInteriorMutable<AVAudioConverter>()
    private let desiredFormat = AVAudioFormat(commonFormat: .pcmFormatInt16, sampleRate: 24000, channels: 1, interleaved: false)!

	/// A stream of errors that occur during the conversation.
	public let errors: AsyncStream<ServerError>

	/// The unique ID of the conversation.
	@MainActor public private(set) var id: String?

	/// The current session for this conversation.
	@MainActor public private(set) var session: Session?

	/// A list of items in the conversation.
	@MainActor public private(set) var entries: [Item] = []

	/// Whether the conversation is currently connected to the server.
	@MainActor public private(set) var connected: Bool = false

	/// Whether the conversation is currently listening to the user's microphone.
    @MainActor public private(set) var isListening: Bool = false {
        didSet {
            notifyDelegates { $0.conversationDidUpdateIsListening(self, isListening: isListening) }
        }
    }

	/// Whether this conversation is currently handling voice input and output.
	@MainActor public private(set) var handlingVoice: Bool = false

	/// Whether the user is currently speaking.
	/// This only works when using the server's voice detection.
	@MainActor public private(set) var isUserSpeaking: Bool = false {
        didSet {
            notifyDelegates { $0.conversationDidUpdateIsUserSpeaking(self, isUserSpeaking: isListening) }
        }
    }

	/// Whether the model is currently speaking.
	@MainActor public private(set) var isPlaying: Bool = false {
        didSet {
            notifyDelegates { $0.conversationDidUpdateIsPlaying(self, isPlaying: isPlaying) }
        }
    }

	/// A list of messages in the conversation.
	/// Note that this doesn't include function call events. To get a complete list, use `entries`.
	@MainActor public var messages: [Item.Message] {
		entries.compactMap { switch $0 {
			case let .message(message): return message
			default: return nil
		} }
	}
    
    @MainActor private var wasListeningBeforeInterruption: Bool = false

	private init(client: RealtimeAPI) {
		self.client = client
		(errors, errorStream) = AsyncStream.makeStream(of: ServerError.self)

		let task = Task.detached { [weak self] in
			guard let self else { return }

			for try await event in client.events {
				await self.handleEvent(event)
			}

			await MainActor.run {
				self.connected = false
			}
		}

		Task { @MainActor in
			self.cancelTask = task.cancel

			client.onDisconnect = { [weak self] in
				guard let self else { return }

				Task { @MainActor in
					self.connected = false
				}
			}

			_keepIsPlayingPropertyUpdated()
		}
	}

	deinit {
		errorStream.finish()

		DispatchQueue.main.asyncAndWait {
			cancelTask?()
			stopHandlingVoice()
		}
	}

	/// Create a new conversation providing an API token and, optionally, a model.
	public convenience init(authToken token: String, model: String = "gpt-4o-realtime-preview") {
		self.init(client: RealtimeAPI.webSocket(authToken: token, model: model))
	}

	/// Create a new conversation that connects using a custom `URLRequest`.
	public convenience init(connectingTo request: URLRequest) {
		self.init(client: RealtimeAPI.webSocket(connectingTo: request))
	}

	/// Wait for the connection to be established
	@MainActor public func waitForConnection() async {
		while true {
			if connected {
				return
			}

			try? await Task.sleep(for: .milliseconds(500))
		}
	}

	/// Execute a block of code when the connection is established
	@MainActor public func whenConnected<E>(_ callback: @Sendable () async throws(E) -> Void) async throws(E) {
		await waitForConnection()
		try await callback()
	}

	/// Make changes to the current session
	/// Note that this will fail if the session hasn't started yet. Use `whenConnected` to ensure the session is ready.
	public func updateSession(withChanges callback: (inout Session) -> Void) async throws {
		guard var session = await session else {
			throw ConversationError.sessionNotFound
		}

		callback(&session)

		try await setSession(session)
	}

	/// Set the configuration of the current session
	public func setSession(_ session: Session) async throws {
		// update endpoint errors if we include the session id
		var session = session
		session.id = nil

		try await client.send(event: .updateSession(session))
	}

	/// Send a client event to the server.
	/// > Warning: This function is intended for advanced use cases. Use the other functions to send messages and audio data.
	public func send(event: ClientEvent) async throws {
		try await client.send(event: event)
	}

	/// Manually append audio bytes to the conversation.
	/// Commit the audio to trigger a model response when server turn detection is disabled.
	/// > Note: The `Conversation` class can automatically handle listening to the user's mic and playing back model responses.
	/// > To get started, call the `startListening` function.
	public func send(audioDelta audio: Data, commit: Bool = false) async throws {
		try await send(event: .appendInputAudioBuffer(encoding: audio))
		if commit { try await send(event: .commitInputAudioBuffer()) }
	}

	/// Send a text message and wait for a response.
	/// Optionally, you can provide a response configuration to customize the model's behavior.
	/// > Note: Calling this function will automatically call `interruptSpeech` if the model is currently speaking.
	public func send(from role: Item.ItemRole, text: String, response: Response.Config? = nil) async throws {
		if await handlingVoice { await interruptSpeech() }

		try await send(event: .createConversationItem(Item(message: Item.Message(id: String(randomLength: 32), from: role, content: [.input_text(text)]))))
		try await send(event: .createResponse(response))
	}
    
    /// Send a  message and wait for a response.
    /// Optionally, you can provide a response configuration to customize the model's behavior.
    /// > Note: Calling this function will automatically call `interruptSpeech` if the model is currently speaking.
    public func send(message: Item.Message, response: Response.Config? = nil) async throws {
        if await handlingVoice { await interruptSpeech() }

        // Use the provided message instead of creating a new one
        try await send(event: .createConversationItem(Item(message: message)))
        try await send(event: .createResponse(response))
    }

	/// Send the response of a function call.
	public func send(result output: Item.FunctionCallOutput) async throws {
		try await send(event: .createConversationItem(Item(with: output)))
	}
    
    // MARK: - Delegate Management
        
    /// Adds a delegate to receive conversation events.
    /// - Parameter delegate: The delegate to add.
    public func addDelegate(_ delegate: ConversationDelegate) {
        delegates.add(delegate)
    }
    
    /// Removes a delegate from receiving conversation events.
    /// - Parameter delegate: The delegate to remove.
    public func removeDelegate(_ delegate: ConversationDelegate) {
        delegates.remove(delegate)
    }
    
    /// Notifies all registered delegates by executing a closure on each.
    /// - Parameter closure: The closure containing the delegate method to call.
    private func notifyDelegates(_ closure: (ConversationDelegate) -> Void) {
        for delegate in delegates.allObjects {
            if let delegate = delegate as? ConversationDelegate {
                closure(delegate)
            }
        }
    }
}

/// Listening/Speaking public API
public extension Conversation {
	/// Start listening to the user's microphone and sending audio data to the model.
	/// This will automatically call `startHandlingVoice` if it hasn't been called yet.
	/// > Warning: Make sure to handle the case where the user denies microphone access.
    @MainActor func startListening() throws {
        guard !isListening else { return }
        if !handlingVoice { try startHandlingVoice() }
        Task.detached {
            self.audioEngine.inputNode.installTap(onBus: 0, bufferSize: 4096, format: self.audioEngine.inputNode.outputFormat(forBus: 0)) { [weak self] buffer, _ in
                self?.processAudioBufferFromUser(buffer: buffer)
            }
        }
        isListening = true
    }

	/// Stop listening to the user's microphone.
	/// This won't stop playing back model responses. To fully stop handling voice conversations, call `stopHandlingVoice`.
    @MainActor func stopListening() {
        guard isListening else { return }
        audioEngine.inputNode.removeTap(onBus: 0)
        isListening = false
    }

    /// Set up the audio engine, install the tap (once) and start handling voice.
    /// This tap is responsible for processing the incoming audio.
    @MainActor func startHandlingVoice() throws {
        guard !handlingVoice else { return }
        
        guard let converter = AVAudioConverter(from: audioEngine.inputNode.outputFormat(forBus: 0), to: desiredFormat) else {
            throw ConversationError.converterInitializationFailed
        }
        userConverter.set(converter)
        
        let audioSession = AVAudioSession.sharedInstance()
        try audioSession.setCategory(.playAndRecord, mode: .voiceChat, options: [.defaultToSpeaker, .allowBluetooth])
        try audioSession.setActive(true)
        
        audioEngine.attach(playerNode)
        audioEngine.connect(playerNode, to: audioEngine.mainMixerNode, format: converter.inputFormat)
        try audioEngine.inputNode.setVoiceProcessingEnabled(true)
        
        audioEngine.prepare()
        do {
            try audioEngine.start()
            handlingVoice = true
            
            // Add observers for interruptions and route changes:
            NotificationCenter.default.addObserver(self,
                                                   selector: #selector(handleRouteChange(_:)),
                                                   name: AVAudioSession.routeChangeNotification,
                                                   object: nil)
            NotificationCenter.default.addObserver(self,
                                                   selector: #selector(handleAudioInterruption(_:)),
                                                   name: AVAudioSession.interruptionNotification,
                                                   object: nil)
        } catch {
            print("Failed to enable audio engine: \(error)")
            audioEngine.disconnectNodeInput(playerNode)
            audioEngine.disconnectNodeOutput(playerNode)
        
            throw error
        }

    }

	/// Interrupt the model's response if it's currently playing.
	/// This lets the model know that the user didn't hear the full response.
	@MainActor func interruptSpeech() {
		if isPlaying,
		   let nodeTime = playerNode.lastRenderTime,
		   let playerTime = playerNode.playerTime(forNodeTime: nodeTime),
		   let itemID = queuedSamples.first
		{
			let audioTimeInMiliseconds = Int((Double(playerTime.sampleTime) / playerTime.sampleRate) * 1000)

			Task {
				do {
					try await client.send(event: .truncateConversationItem(forItem: itemID, atAudioMs: audioTimeInMiliseconds))
				} catch {
					print("Failed to send automatic truncation event: \(error)")
				}
			}
		}

		playerNode.stop()
		queuedSamples.clear()
	}

	/// Stop playing audio responses from the model and listening to the user's microphone.
    @MainActor func stopHandlingVoice() {
        guard handlingVoice else { return }
        
        // Remove the observers since we're stopping voice handling
        NotificationCenter.default.removeObserver(self,
                                                  name: AVAudioSession.routeChangeNotification,
                                                  object: nil)
        NotificationCenter.default.removeObserver(self,
                                                  name: AVAudioSession.interruptionNotification,
                                                  object: nil)
    
        // Remove the tap if it's still installed.
        audioEngine.inputNode.removeTap(onBus: 0)
        audioEngine.stop()
        audioEngine.disconnectNodeInput(playerNode)
        audioEngine.disconnectNodeOutput(playerNode)
        
        try? AVAudioSession.sharedInstance().setActive(false)
            
        isListening = false
        handlingVoice = false
    }
    
    @objc private func handleRouteChange(_ notification: Notification) {
        guard let userInfo = notification.userInfo,
              let reasonValue = userInfo[AVAudioSessionRouteChangeReasonKey] as? UInt,
              let reason = AVAudioSession.RouteChangeReason(rawValue: reasonValue) else { return }
        
        print("Audio route changed: \(reason)")
        
        switch reason {
        case .oldDeviceUnavailable:
            // Use your higher-level API to restart listening.
            print("Old device unavailable. Restarting listening.")
            Task { @MainActor in
                // Stop and restart listening if it was active.
                if self.isListening {
                    self.stopListening()
                    do {
                        try self.startListening()
                    } catch {
                        print("Error restarting listening after route change: \(error)")
                    }
                }
            }
            
        default:
            break
        }
    }

    @objc private func handleAudioInterruption(_ notification: Notification) {
        guard let userInfo = notification.userInfo,
              let typeValue = userInfo[AVAudioSessionInterruptionTypeKey] as? UInt,
              let interruptionType = AVAudioSession.InterruptionType(rawValue: typeValue) else {
            return
        }
        
        switch interruptionType {
        case .began:
            print("Audio interruption began.")
            Task { @MainActor in
                // Save current listening state before stopping
                self.wasListeningBeforeInterruption = self.isListening
                if self.isListening {
                    self.stopListening()
                }
                // Pause playback if it is active
                if self.isPlaying {
                    self.playerNode.pause()
                }
                // Pause the audio engine if it's running
                if self.audioEngine.isRunning {
                    self.audioEngine.pause()
                }
            }
            
        case .ended:
            print("Audio interruption ended.")
            let optionsValue = userInfo[AVAudioSessionInterruptionOptionKey] as? UInt
            let options = AVAudioSession.InterruptionOptions(rawValue: optionsValue ?? 0)
            
            if options.contains(.shouldResume) {
                Task { @MainActor in
                    do {
                        // Restart the audio engine
                        try self.audioEngine.start()
                        
                        // Resume listening if we were listening before the interruption
                        if self.wasListeningBeforeInterruption {
                            try self.startListening()
                            self.wasListeningBeforeInterruption = false
                        }
                        
                        // Resume audio playback if needed
                        if self.isPlaying {
                            self.playerNode.play()
                        }
                    } catch {
                        print("Failed to restart audio engine after interruption: \(error)")
                    }
                }
            }
            
        @unknown default:
            break
        }
    }
}

/// Event handling private API
private extension Conversation {
	@MainActor func handleEvent(_ event: ServerEvent) {
        Task {
            // Notify delegates about the received event
            await MainActor.run {
                self.notifyDelegates { $0.conversationDidReceiveEvent(self, event: event) }
            }
        }

		switch event {
			case let .error(event):
				errorStream.yield(event.error)
			case let .sessionCreated(event):
				connected = true
				session = event.session
			case let .sessionUpdated(event):
				session = event.session
			case let .conversationCreated(event):
				id = event.conversation.id
			case let .conversationItemCreated(event):
				entries.append(event.item)
			case let .conversationItemDeleted(event):
				entries.removeAll { $0.id == event.itemId }
			case let .conversationItemInputAudioTranscriptionCompleted(event):
				updateEvent(id: event.itemId) { message in
					guard case let .input_audio(audio) = message.content[event.contentIndex] else { return }

					message.content[event.contentIndex] = .input_audio(.init(audio: audio.audio, transcript: event.transcript))
				}
			case let .conversationItemInputAudioTranscriptionFailed(event):
				errorStream.yield(event.error)
			case let .responseContentPartAdded(event):
				updateEvent(id: event.itemId) { message in
					message.content.insert(.init(from: event.part), at: event.contentIndex)
				}
			case let .responseContentPartDone(event):
				updateEvent(id: event.itemId) { message in
					message.content[event.contentIndex] = .init(from: event.part)
				}
			case let .responseTextDelta(event):
				updateEvent(id: event.itemId) { message in
					guard case let .text(text) = message.content[event.contentIndex] else { return }

					message.content[event.contentIndex] = .text(text + event.delta)
				}
			case let .responseTextDone(event):
				updateEvent(id: event.itemId) { message in
					message.content[event.contentIndex] = .text(event.text)
				}
			case let .responseAudioTranscriptDelta(event):
				updateEvent(id: event.itemId) { message in
					guard case let .audio(audio) = message.content[event.contentIndex] else { return }

					message.content[event.contentIndex] = .audio(.init(audio: audio.audio, transcript: (audio.transcript ?? "") + event.delta))
				}
			case let .responseAudioTranscriptDone(event):
				updateEvent(id: event.itemId) { message in
					guard case let .audio(audio) = message.content[event.contentIndex] else { return }

					message.content[event.contentIndex] = .audio(.init(audio: audio.audio, transcript: event.transcript))
				}
			case let .responseAudioDelta(event):
				updateEvent(id: event.itemId) { message in
					guard case let .audio(audio) = message.content[event.contentIndex] else { return }

					if handlingVoice { queueAudioSample(event) }
					message.content[event.contentIndex] = .audio(.init(audio: audio.audio + event.delta, transcript: audio.transcript))
				}
			case let .responseFunctionCallArgumentsDelta(event):
				updateEvent(id: event.itemId) { functionCall in
					functionCall.arguments.append(event.delta)
				}
			case let .responseFunctionCallArgumentsDone(event):
				updateEvent(id: event.itemId) { functionCall in
					functionCall.arguments = event.arguments
				}
			case .inputAudioBufferSpeechStarted:
				isUserSpeaking = true
				if handlingVoice { interruptSpeech() }
			case .inputAudioBufferSpeechStopped:
				isUserSpeaking = false
            case let .responseOutputItemDone(event):
                updateEvent(id: event.item.id) { message in
                    guard case let .message(newMessage) = event.item else { return }
                    
                    message = newMessage
                }
			default:
				return
		}
	}

	@MainActor
	func updateEvent(id: String, modifying closure: (inout Item.Message) -> Void) {
		guard let index = entries.firstIndex(where: { $0.id == id }), case var .message(message) = entries[index] else {
			return
		}

		closure(&message)

		entries[index] = .message(message)
	}

	@MainActor
	func updateEvent(id: String, modifying closure: (inout Item.FunctionCall) -> Void) {
		guard let index = entries.firstIndex(where: { $0.id == id }), case var .functionCall(functionCall) = entries[index] else {
			return
		}

		closure(&functionCall)

		entries[index] = .functionCall(functionCall)
	}
}

/// Audio processing private API
private extension Conversation {
	private func queueAudioSample(_ event: ServerEvent.ResponseAudioDeltaEvent) {
        guard let buffer = AVAudioPCMBuffer.fromData(event.delta, format: desiredFormat) else {
            print("Failed to create audio buffer.")
            return
        }
        
        guard let converter = apiConverter.lazy({ AVAudioConverter(from: buffer.format, to: playerNode.outputFormat(forBus: 0)) }) else {
            print("Failed to create audio converter.")
            return
        }

		let outputFrameCapacity = AVAudioFrameCount(ceil(converter.outputFormat.sampleRate / buffer.format.sampleRate) * Double(buffer.frameLength))

		guard let sample = convertBuffer(buffer: buffer, using: converter, capacity: outputFrameCapacity) else {
			print("Failed to convert buffer.")
			return
		}

        DispatchQueue.main.async {
            self.queuedSamples.push(event.itemId)
        }

        playerNode.scheduleBuffer(sample, at: nil, completionCallbackType: .dataPlayedBack) { [weak self] _ in
            guard let self = self else { return }
            DispatchQueue.main.async {
                if !self.queuedSamples.isEmpty {
                    self.queuedSamples.popFirst()
                }
                if self.queuedSamples.isEmpty {
                    self.playerNode.pause()
                }
            }
        }

        // Ensure the audio engine is running before playing.
        if !audioEngine.isRunning {
            do {
                try audioEngine.start()
            } catch {
                print("Failed to start audio engine: \(error)")
            }
        }
        playerNode.play()

	}

	private func processAudioBufferFromUser(buffer: AVAudioPCMBuffer) {
		let ratio = desiredFormat.sampleRate / buffer.format.sampleRate

        
        guard let converter = AVAudioConverter(from: buffer.format, to: desiredFormat) else {
            print("Failed to create converter for input format: \(buffer.format)")
            return
        }
        guard let convertedBuffer = convertBuffer(buffer: buffer, using: converter, capacity: AVAudioFrameCount(Double(buffer.frameLength) * ratio)) else {
            print("Buffer conversion failed.")
            return
        }
        
		guard let sampleBytes = convertedBuffer.audioBufferList.pointee.mBuffers.mData else { return }
		let audioData = Data(bytes: sampleBytes, count: Int(convertedBuffer.audioBufferList.pointee.mBuffers.mDataByteSize))

		Task {
			try await send(audioDelta: audioData)
		}
	}

	private func convertBuffer(buffer: AVAudioPCMBuffer, using converter: AVAudioConverter, capacity: AVAudioFrameCount) -> AVAudioPCMBuffer? {
		if buffer.format == converter.outputFormat {
			return buffer
		}

		guard let convertedBuffer = AVAudioPCMBuffer(pcmFormat: converter.outputFormat, frameCapacity: capacity) else {
			print("Failed to create converted audio buffer.")
			return nil
		}

		var error: NSError?
		var allSamplesReceived = false

		let status = converter.convert(to: convertedBuffer, error: &error) { _, outStatus in
			if allSamplesReceived {
				outStatus.pointee = .noDataNow
				return nil
			}

			allSamplesReceived = true
			outStatus.pointee = .haveData
			return buffer
		}

		if status == .error {
			if let error = error {
				print("Error during conversion: \(error.localizedDescription)")
			}
			return nil
		}

		return convertedBuffer
	}
}

// Other private methods
extension Conversation {
	/// This hack is required because relying on `queuedSamples.isEmpty` directly crashes the app.
	/// This is because updating the `queuedSamples` array on a background thread will trigger a re-render of any views that depend on it on that thread.
	/// So, instead, we observe the property and update `isPlaying` on the main actor.
	private func _keepIsPlayingPropertyUpdated() {
		withObservationTracking { _ = queuedSamples.isEmpty } onChange: {
			Task { @MainActor in
				self.isPlaying = !self.queuedSamples.isEmpty
			}

			self._keepIsPlayingPropertyUpdated()
		}
	}
}
