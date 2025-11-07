#!/usr/bin/env bash
# wait-for-spark.sh
while ! curl -s http://spark-master:8080/ | grep -q "Spark"; do
  echo "Waiting for Spark Master..."
  sleep 2
done
echo "Spark is up – starting Streamlit"
exec "$@"
