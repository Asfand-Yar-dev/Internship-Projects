FROM ubuntu

RUN apt-get update
RUN apt-get install -y curl
RUN curl -sl https://deb.nodesource.com/setup_18.x | bash 
RUn apt-get upgrade -y
RUN apt-get install -y nodejs

COPY package.json package.json
COPY package-lock.json package-lock.json
COPY main.js main.js

RUN npm install


EntryPoint ["node", "main.js"]