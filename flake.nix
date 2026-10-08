{
  description = "Idiolect — a local voice cloning CLI";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  outputs = { nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
    in {
      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [ python312 uv ffmpeg-headless sox libsndfile pipewire ];
        LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath (with pkgs; [
          stdenv.cc.cc.lib zlib glib libsndfile
        ]);
        IDIOLECT_NIX = "1";
        UV_PYTHON_DOWNLOADS = "never";
        UV_PYTHON = "${pkgs.python312}/bin/python3.12";
        SSL_CERT_FILE = "${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt";
        PYTHONNOUSERSITE = "1";
      };
    };
}
