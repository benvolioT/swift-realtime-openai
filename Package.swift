// swift-tools-version: 6.0

import PackageDescription

let package = Package(
	name: "SwiftRealtimeOpenAI",
	platforms: [
		.iOS(.v17),
		.tvOS(.v17),
		.macOS(.v14),
		.watchOS(.v10),
		.visionOS(.v1),
		.macCatalyst(.v17),
	],
	products: [
		.library(name: "SwiftRealtimeOpenAI", type: .static, targets: ["SwiftRealtimeOpenAI"]),
	],
	dependencies: [
		.package(url: "https://github.com/stasel/WebRTC.git", branch: "latest"),
	],
	targets: [
		.target(name: "OpenAI", dependencies: ["WebRTC"], path: "./src"),
	]
)
