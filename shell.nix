{ pkgs ? import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/nixos-25.05.tar.gz")
    { config.allowUnfree = true; } }:

# Pinned to nixos-25.05 for python3.10

let
  dynamic-libs = with pkgs; [
    stdenv.cc.cc.lib # Provides libstdc++.so.6 (crucial for TensorFlow/Keras/Numpy)
    zlib             # Used for data compression
    glib             # Common dependency for underlying C libraries
    freetype         # Needed by wordcloud to render fonts
    libjpeg          # Needed by image processing libraries
    libGL            # Needed by opencv-python and mediapipe
    libsndfile       # Needed by librosa (via soundfile)
  ];
in
pkgs.mkShell {
  packages = with pkgs; [
    python310

    gcc
    gnumake

    ngrok
    wget
    ffmpeg
  ];

  shellHook = ''
    export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath dynamic-libs}:$LD_LIBRARY_PATH"

    if [ -d /run/opengl-driver/lib ]; then
      export LD_LIBRARY_PATH="/run/opengl-driver/lib:$LD_LIBRARY_PATH"
    else
      echo "warning: /run/opengl-driver/lib not found -- GPU may be unavailable"
    fi

    if [ ! -d ".venv" ]; then
      echo "Creating venv..."
      python3.10 -m venv .venv
    fi

    # Activate the environment
    source .venv/bin/activate

    # Install the requirements
    if [ -f "requirements.txt" ]; then
      echo "Downloading requirements.txt..."
      # Using --prefer-binary prevents pip from trying to compile heavy ML packages from source
      pip install --prefer-binary -r requirements.txt
    fi
    
    echo "Environment done"
  '';
}
