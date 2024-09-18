FROM python:3.10

WORKDIR /leaffliction

COPY requirements.txt /leaffliction/requirements.txt

RUN pip install -r requirements.txt

RUN echo 'alias norminette="flake8"' >> /root/.bashrc

CMD ["top", "-b"]