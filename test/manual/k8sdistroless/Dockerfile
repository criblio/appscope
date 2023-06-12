FROM gcc:12 AS build-env
COPY . /app
WORKDIR /app
RUN cc main.c -o hello_server

FROM gcr.io/distroless/cc
COPY --from=build-env /app /app
WORKDIR /app
CMD ["./hello_server"]
