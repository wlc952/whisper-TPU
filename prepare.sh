sudo apt install unzip
pip3 install --upgrade pip
pip3 install --no-cache-dir -r python/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

if ! python3 -c "import sophon.sail" &> /dev/null; then
    echo "Sophon SAIL is not found, installing..."
    pip3 install https://modelscope.cn/models/wlc952/aigchub_models/resolve/master/sophon_arm-3.8.0-py3-none-any.whl
fi
