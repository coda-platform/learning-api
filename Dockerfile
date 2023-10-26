FROM node:16

WORKDIR /usr/src/app
COPY ./ ./

RUN rm -r -f ./build
RUN npm install
RUN npm run build

# Make build footprint version for easier debugging.
RUN rm ./version.txt
RUN openssl rand -hex 12 > version.txt

# Install local packages for running server & tensorflowjs on host.
RUN npm rebuild @tensorflow/tfjs-node --build-addon-from-source
RUN npm install pm2 -g

EXPOSE 8084
CMD ["pm2-runtime", "build/index.js", "--node-args=\"--max-old-space-size=8192\""]
