FROM python:3.10
RUN pip install --upgrade pip
RUN pip install tensorboard

# Warning in logs is fine: https://github.com/lanpa/tensorboardX/issues/602#issuecomment-693246519
CMD tensorboard --logdir=/tb_folder --host 0.0.0.0 --port 6006
