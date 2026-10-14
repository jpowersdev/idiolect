{
  description = "Local voice cloning from the terminal";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, pyproject-nix, uv2nix, pyproject-build-systems }:
    let
      inherit (nixpkgs) lib;
      # Linux builds from the same lockfile, but its CUDA wheels need patching
      # against the system driver first.
      systems = [ "aarch64-darwin" ];
      forAllSystems = f: lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});

      workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./engine; };

      # The engine and its locked dependencies, from PyPI wheels in engine/uv.lock.
      engineFor = pkgs:
        let
          pythonSet = (pkgs.callPackage pyproject-nix.build.packages { python = pkgs.python312; }).overrideScope
            (lib.composeManyExtensions [
              pyproject-build-systems.overlays.wheel
              (workspace.mkPyprojectOverlay { sourcePreference = "wheel"; })
              (final: prev: {
                # Only published as a source distribution.
                sox = prev.sox.overrideAttrs (old: {
                  nativeBuildInputs = old.nativeBuildInputs ++ final.resolveBuildSystem { setuptools = [ ]; };
                });
              })
            ]);
        in
        pythonSet.mkVirtualEnv "idiolect-engine-${version}" workspace.deps.default;

      version = (lib.importJSON ./package.json).version;

      cliFor = pkgs: engine:
        let
          src = lib.fileset.toSource {
            root = ./.;
            fileset = lib.fileset.unions [ ./package.json ./bun.lock ./tsconfig.json ./src ];
          };
          # Bun fetches the locked dependencies; update the hash when bun.lock changes.
          nodeModules = pkgs.stdenvNoCC.mkDerivation {
            pname = "idiolect-node-modules";
            inherit version src;
            nativeBuildInputs = [ pkgs.bun ];
            dontConfigure = true;
            buildPhase = ''
              runHook preBuild
              export HOME=$TMPDIR
              bun install --frozen-lockfile --production --ignore-scripts --no-progress
              runHook postBuild
            '';
            installPhase = "cp -R node_modules $out";
            dontFixup = true;
            outputHashMode = "recursive";
            outputHashAlgo = "sha256";
            outputHash = "sha256-d2tsiEYZ/TDP8C23XcpoZE8rxV/Ulnm4RjGPYGCSnls=";
          };
        in
        pkgs.stdenvNoCC.mkDerivation {
          pname = "idiolect";
          inherit version src;
          nativeBuildInputs = [ pkgs.bun pkgs.makeWrapper ];
          dontConfigure = true;
          buildPhase = ''
            runHook preBuild
            # Copied: Bun resolves packages from their real paths.
            cp -R ${nodeModules} node_modules
            bun build src/main.ts --target=bun --outfile=main.js
            runHook postBuild
          '';
          installPhase = ''
            runHook preInstall
            install -Dm644 main.js $out/lib/idiolect/main.js
            # Only the engine's entry point joins PATH, not its Python environment.
            mkdir -p $out/libexec/idiolect
            ln -s ${engine}/bin/idiolect-engine $out/libexec/idiolect/idiolect-engine
            makeWrapper ${lib.getExe pkgs.bun} $out/bin/idiolect \
              --add-flags $out/lib/idiolect/main.js \
              --prefix PATH : ${lib.makeBinPath [ pkgs.sox ]}:$out/libexec/idiolect
            runHook postInstall
          '';
          meta = {
            description = "Local voice cloning from the terminal";
            mainProgram = "idiolect";
            platforms = systems;
          };
        };
    in
    {
      packages = forAllSystems (pkgs: rec {
        engine = engineFor pkgs;
        idiolect = cliFor pkgs engine;
        default = idiolect;
      });

      devShells = forAllSystems (pkgs: {
        # For working on Idiolect: uv manages engine/.venv, Bun manages node_modules.
        default = pkgs.mkShell {
          packages = with pkgs; [ bun python312 uv sox ];
          UV_PYTHON_DOWNLOADS = "never";
          UV_PYTHON = lib.getExe pkgs.python312;
          shellHook = ''
            export PATH="$PWD/engine/.venv/bin:$PATH"
          '';
        };
      });
    };
}
