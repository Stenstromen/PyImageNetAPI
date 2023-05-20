# PyImageNetAPI

Python Flask API Docker image using Keras MobileNetV2 pre-trained model.

Post image, get prediction.

## Build
```
docker build -t imagenetapi:latest .
```

## Run
```
docker run --rm -d \
--name pyimagenetapi \
-v $PWD/models:/root/.keras/models \
-p 5000:5000 \
imagenetapi:latest
```

## Test
```
curl -s -X POST -F "file=@cat.jpg" "http://localhost:5000/predict?limit=3"|jq
```