FROM python:3.13
COPY . /app/
WORKDIR /app/
RUN pip install -r requirements.txt
# Install all plugins from `additional-requirements.txt` files in the plugins directory.
RUN for file in plugins/*/additional-requirements.txt; do pip install -r $file; done
CMD python syncer.py