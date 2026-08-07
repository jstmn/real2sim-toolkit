docker rm -f foundationpose

DIR=$(pwd)/../

docker run \
  --gpus all \
  --env NVIDIA_DISABLE_REQUIRE=1 \
  -it \
  --network=host \
  --name foundationpose \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -v "$DIR:$DIR" \
  -v /home:/home \
  -v /mnt:/mnt \
  -v /tmp:/tmp \
  --ipc=host \
  foundationpose:latest \
  bash -c "cd $DIR && bash"
