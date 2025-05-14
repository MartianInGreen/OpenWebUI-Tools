FROM alpine:latest
RUN apk add --no-cache python3 py3-pip
WORKDIR /app

# Define environment variables
ENV APP_PORT=9020
ENV DEBUG=false
ENV API_KEY=some_key
ENV EXTERNAL_URL=https://api.some_url.com

# Write environment variables to file
RUN echo "APP_PORT=$APP_PORT" > .env
RUN echo "DEBUG=$DEBUG" >> .env
RUN echo "API_KEY=$API_KEY" >> .env
RUN echo "EXTERNAL_URL=$EXTERNAL_URL" >> .env

COPY . /app
RUN pip3 install -r requirements.txt
