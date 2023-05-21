# PyImageNetAPI

Python Flask API Docker image using Keras MobileNetV3 pre-trained model.

Post image, get prediction.

## Build
```
docker build -t imagenetapi:latest .
```

## Run
```
docker run --rm -d \
--name pyimagenetapi \
-e CORS_ORIGINS="*" \
-e AUTHORIZATION_KEY="123" \
-v $PWD/models:/root/.keras/models \
-p 8080:80 \
imagenetapi:latest
```

## Test
```
curl -s -X POST -F "file=@cat.jpg" "http://localhost:8080/predict?limit=3" | jq
```