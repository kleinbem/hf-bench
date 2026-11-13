{
  lib,
  stdenvNoCC,
  makeWrapper,
  python3,
  llama-cpp,
  openssh,
  util-linux,
  runCommand,
}:

let
  pyEnv = python3.withPackages (ps: [
    ps.gguf
    ps.huggingface-hub
    ps.numpy
    ps.pyarrow
    ps.safetensors
    ps.sentencepiece
    ps.torchWithoutCuda
    ps.transformers
  ]);

  runtimeDeps = [
    llama-cpp
    openssh
  ]
  ++ lib.optionals stdenvNoCC.hostPlatform.isLinux [
    util-linux
  ];
in
stdenvNoCC.mkDerivation (finalAttrs: {
  pname = "hf-bench";
  version = "0.1.0";

  src = lib.fileset.toSource {
    root = ../python;
    fileset = ../python;
  };

  nativeBuildInputs = [ makeWrapper ];

  installPhase = ''
    runHook preInstall

    mkdir -p "$out/lib/python" "$out/bin"
    cp -r hf_bench "$out/lib/python/"

    makeWrapper "${pyEnv}/bin/python3" "$out/bin/convert-hf-to-gguf" \
      --add-flags "${llama-cpp.src}/convert_hf_to_gguf.py"

    makeWrapper "${pyEnv}/bin/python3" "$out/bin/hf-bench" \
      --prefix PYTHONPATH : "$out/lib/python" \
      --set CONVERT_HF_TO_GGUF "$out/bin/convert-hf-to-gguf" \
      --prefix PATH : "${lib.makeBinPath runtimeDeps}:$out/bin" \
      --add-flags "-m hf_bench.cli"

    runHook postInstall
  '';

  passthru.tests.unit =
    runCommand "${finalAttrs.pname}-unit-tests"
      {
        nativeBuildInputs = [ pyEnv ];
      }
      ''
        PYTHONPATH="${../python}" python3 -m unittest discover -s "${../python}/tests" -v
        "${finalAttrs.finalPackage}/bin/hf-bench" --help >/dev/null
        touch "$out"
      '';

  meta = {
    description = "CLI orchestrator for edge GGUF quantization, KL-divergence, and hardware benchmarks";
    homepage = "https://github.com/kleinbem/hf-bench";
    license = lib.licenses.mit;
    maintainers = [ lib.maintainers.kleinbem ];
    mainProgram = "hf-bench";
  };
})
