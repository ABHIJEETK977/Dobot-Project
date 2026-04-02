1) Activate the venv that contains Hailo + OpenCV + Vosk dependencies:
   source /home/dobot/Documents/Dobot/Ishan/better/venv_hailo_rpi_examples/bin/activate

2) Run the script with:
   python working_vcv.py \
     --hef-path resources/fruit_dobot_v1.hef \
     --input /dev/video0 \
     --labels-json resources/fruit_dobot_v1.json
