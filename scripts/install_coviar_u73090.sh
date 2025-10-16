#!/usr/bin/env bash
set -euo pipefail
echo ">>> install_coviar_u73090.sh: build FFmpeg 3.4.8 and coviar on this machine"

# 推荐在conda环境里执行（如果没有会跳过）
conda activate mccar || true

sudo apt-get update
sudo apt-get install -y yasm nasm gcc g++ make pkg-config git

mkdir -p "$HOME/build" && cd "$HOME/build"
rm -rf ffmpeg-coviar-src
git clone https://github.com/FFmpeg/FFmpeg.git ffmpeg-coviar-src
cd ffmpeg-coviar-src
git checkout n3.4.8
./configure --prefix="$HOME/ffmpeg-coviar" --enable-shared --enable-pic --enable-yasm --disable-inline-asm
make -j"$(nproc)"
make install

export PKG_CONFIG_PATH="$HOME/ffmpeg-coviar/lib/pkgconfig"
export LD_LIBRARY_PATH="$HOME/ffmpeg-coviar/lib:$LD_LIBRARY_PATH"

cd /tmp
rm -rf pytorch-coviar
git clone https://github.com/chaoyuaw/pytorch-coviar.git
cd pytorch-coviar/data_loader
CFLAGS="$(pkg-config --cflags libavformat libavcodec libavutil libswscale libswresample)" \
LDFLAGS="$(pkg-config --libs   libavformat libavcodec libavutil libswscale libswresample)" \
python setup.py build_ext --inplace
pip install .

python - <<'PY'
from coviar import get_num_frames, load
print("✅ coviar import OK")
PY
echo ">>> done."
