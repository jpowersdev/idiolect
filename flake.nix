{
  description = "Local voice cloning from the terminal";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      inherit (nixpkgs) lib;
      # effect-torch's Metal backend; Linux follows with its CUDA backend.
      systems = [ "aarch64-darwin" ];
      forAllSystems = f: lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});

      version = (lib.importJSON ./package.json).version;

      idiolectFor = pkgs:
        let
          src = lib.fileset.toSource {
            root = ./.;
            fileset = lib.fileset.unions [ ./package.json ./bun.lock ./tsconfig.json ./src ];
          };
          # Bun fetches the locked dependencies, effect-torch's prebuilt native addons included;
          # update the hash when bun.lock changes.
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
            outputHash = lib.fakeHash;
          };
        in
        pkgs.stdenvNoCC.mkDerivation {
          pname = "idiolect";
          inherit version src;
          nativeBuildInputs = [ pkgs.makeWrapper ];
          dontConfigure = true;
          dontBuild = true;
          installPhase = ''
            runHook preInstall
            # Run from source, as the npm package does: the CLI starts src/engine/main.ts
            # beside it with the same Bun.
            mkdir -p $out/lib/idiolect
            cp -R package.json tsconfig.json src $out/lib/idiolect/
            cp -R ${nodeModules} $out/lib/idiolect/node_modules
            makeWrapper ${lib.getExe pkgs.bun} $out/bin/idiolect \
              --add-flags $out/lib/idiolect/src/main.ts \
              --prefix PATH : ${lib.makeBinPath [ pkgs.sox ]}
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
        idiolect = idiolectFor pkgs;
        default = idiolect;
      });

      devShells = forAllSystems (pkgs: {
        default = pkgs.mkShell {
          packages = with pkgs; [ bun sox ];
        };
      });
    };
}
