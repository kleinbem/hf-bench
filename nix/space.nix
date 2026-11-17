{
  lib,
  buildNpmPackage,
  nodejs,
  runCommand,
  caddy,
  curl,
  jq,
  writeText,
}:

let
  spaceReadme = writeText "hf-space-readme.md" ''
    ---
    title: Edge & Local LLM Pareto Explorer
    emoji: ⚡
    colorFrom: blue
    colorTo: indigo
    sdk: static
    pinned: true
    license: mit
    short_description: GGUF KL-divergence, tok/s, Joules/tok & 8 GiB Edge KV sizer
    ---

    # Edge & Local GGUF Pareto Frontier Explorer

    Interactive explorer for calibrated `llama.cpp` `imatrix` GGUF quantizations, exact FP16 logit KL-divergence, and hardware power/throughput telemetry across 8 GiB NVIDIA Jetson Orin Nano (`sm_87` CUDA 13) and x86_64 AVX2.

    - **Dataset**: [`kleinbem/edge-llm-benchmarks`](https://huggingface.co/datasets/kleinbem/edge-llm-benchmarks)
    - **Source & Reproducible Nix Flake**: [`github.com/kleinbem/hf-bench`](https://github.com/kleinbem/hf-bench)
  '';
in
buildNpmPackage (finalAttrs: {
  pname = "hf-bench-space";
  version = "0.1.0";

  src = lib.fileset.toSource {
    root = ../.;
    fileset = lib.fileset.unions [
      ../package.json
      ../package-lock.json
      ../tsconfig.json
      ../vite.config.ts
      ../src
      ../static
      ../test
    ];
  };

  npmDepsHash = "sha256-QJh5p6k3SDdHkLc1XZhYjFqpAuK9M62maGy45DrHQek=";
  # no-link: standard buildNpmPackage option when unpacking git lockfile entries (@kleinbem/web-config).
  makeCacheWritable = true;

  strictDeps = true;
  __structuredAttrs = true;

  doCheck = true;

  checkPhase = ''
    runHook preCheck
    npm test
    npm run check
    runHook postCheck
  '';

  installPhase = ''
    runHook preInstall
    mkdir -p "$out"
    cp -r build/* "$out/"
    cp "${spaceReadme}" "$out/README.md"
    runHook postInstall
  '';

  doInstallCheck = true;

  installCheckPhase = ''
    runHook preInstallCheck

    for expected in index.html 200.html README.md data/benchmarks.json _app; do
      if [ ! -e "$out/$expected" ]; then
        echo "missing expected Space output: $expected" >&2
        exit 1
      fi
    done

    if grep -rn '/nix/store/' "$out"; then
      echo "unexpected /nix/store reference leaked into static Space bundle" >&2
      exit 1
    fi

    runHook postInstallCheck
  '';

  passthru.tests.smoke =
    runCommand "${finalAttrs.pname}-smoke-test"
      {
        nativeBuildInputs = [
          caddy
          curl
          jq
        ];
      }
      ''
        sock="$TMPDIR/caddy.sock"
        cat > Caddyfile <<EOF
        {
          admin off
          auto_https off
        }
        http:// {
          bind unix/$sock
          root * ${finalAttrs.finalPackage}
          try_files {path} /200.html
          file_server
        }
        EOF

        HOME="$TMPDIR" XDG_DATA_HOME="$TMPDIR" XDG_CONFIG_HOME="$TMPDIR" \
          caddy run --config Caddyfile >"$TMPDIR/caddy.log" 2>&1 &
        caddy_pid=$!
        trap 'kill "$caddy_pid" 2>/dev/null || true' EXIT

        ready=0
        for _ in $(seq 1 50); do
          if curl -fsS --unix-socket "$sock" http://localhost/ >/dev/null 2>&1; then
            ready=1
            break
          fi
          sleep 0.1
        done
        if [ "$ready" -ne 1 ]; then
          cat "$TMPDIR/caddy.log" >&2
          echo "caddy failed to start on $sock" >&2
          exit 1
        fi

        curl -fsS --unix-socket "$sock" http://localhost/ | grep -q "sveltekit"
        curl -fsS --unix-socket "$sock" http://localhost/data/benchmarks.json | jq -e '.models | length >= 2' >/dev/null

        touch "$out"
      '';

  meta = {
    description = "Interactive Edge & Local GGUF Pareto Frontier & 8 GiB Unified-Memory Sizer";
    homepage = "https://github.com/kleinbem/hf-bench";
    license = lib.licenses.mit;
    maintainers = [ lib.maintainers.kleinbem ];
    inherit (nodejs.meta) platforms;
  };
})
