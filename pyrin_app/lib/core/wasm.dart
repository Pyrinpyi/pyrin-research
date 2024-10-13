//
// import "dart:convert";
// import "dart:typed_data";
//
// import "package:wasm_run_flutter/wasm_run_flutter.dart";
//
// import 'package:flutter/services.dart' show rootBundle;
//
// Future<Uint8List> loadAssetBytes(String path) async {
//   ByteData data = await rootBundle.load(path);
//   return data.buffer.asUint8List();
// }
//
// Future<void> init() async
// {
//     // pyrin-wallet_bg.wasm
//
//     const base64Binary =
//         'AGFzbQEAAAABBwFgAn9/AX8DAgEABwcBA2FkZAAACgkBBwAgACABagsAEARuYW1lAgkBAAIAAWEBAWI=';
//     // final Uint8List binary = base64Decode(base64Binary) as Uint8List;
//
//     Uint8List binary = await loadAssetBytes("assets/pyrin_flutter_bg.wasm");
//
//     final WasmModule module = await compileWasmModule(binary);
//
//     // final WasmModule module = await compileWasmModule(
//     //     binary as Uint8List,
//     //     config: const ModuleConfig(
//     //         wasmi: ModuleConfigWasmi(),
//     //         wasmtime: ModuleConfigWasmtime(),
//     //     ),
//     // );
//     final List<WasmModuleExport> exports = module.getExports();
//
//     for (var export in exports) {
//       print("export: ${export.name}");
//     }
//
//     // WasiConfig? wasiConfig;
//     // final WasmInstanceBuilder builder = module.builder(wasiConfig: wasiConfig);
//
//     // final WasmInstance instance = await builder.build();
//     // final WasmFunction kaspaToSompi = instance.getFunction("kaspaToSompi")!;
//     //
//     // final List<Object?> result = kaspaToSompi(["0.001"]);
//     //
//     // print("result: $result");
//
//     // assert(
//     // exports.first.toString() ==
//     //     const WasmModuleExport('Wallet', WasmExternalKind.function).toString(),
//     // );
//     // final List<WasmModuleImport> imports = module.getImports();
//     // assert(imports.isEmpty);
//     //
//     // // configure wasi
//     // WasiConfig? wasiConfig;
//     // final WasmInstanceBuilder builder = module.builder(wasiConfig: wasiConfig);
//     //
//     // // create external
//     // // builder.createTable
//     // // builder.createGlobal
//     // // builder.createMemory
//     //
//     // // Add imports
//     // // builder.addImport(moduleName, name, value);
//     //
//     // final WasmInstance instance = await builder.build();
//     // final WasmFunction add = instance.getFunction('add')!;
//     //
//     // final List<ValueTy?> params = add.params;
//     // assert(params.length == 2);
//     //
//     // final WasmRuntimeFeatures runtime = await wasmRuntimeFeatures();
//     // if (!runtime.isBrowser) {
//     //   // Types are not supported in browser
//     //   assert(params.every((t) => t == ValueTy.i32));
//     //   assert(add.results!.length == 1);
//     //   assert(add.results!.first == ValueTy.i32);
//     // }
//     //
//     // final List<Object?> result = add([1, 4]);
//     // assert(result.length == 1);
//     // assert(result.first == 5);
//     //
//     // print("result.first ${result.first}");
//     // print("result.last: ${result.last}");
//     //
//     // final resultInner = add.inner(-1, 8) as int;
//     // assert(resultInner == 7);
// }
