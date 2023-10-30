rm -r -f ./build

docker build -t coda-learning-api:dev .

docker tag coda19-learning-api:dev coda19/coda19-learning-api:dev
docker push coda19/coda19-learning-api:dev
echo "Finished running script sleeping 30s"
sleep 30