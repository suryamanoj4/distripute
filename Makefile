.PHONY: proto test clean

proto:
	uv run python3 -m grpc_tools.protoc \
		-I proto \
		--python_out=distripute/grpc \
		--grpc_python_out=distripute/grpc \
		proto/distripute.proto
	# fix relative import in generated grpc file
	sed -i 's/^import distripute_pb2/from . import distripute_pb2/' distripute/grpc/distripute_pb2_grpc.py

test:
	uv run pytest tests/ -v

clean:
	rm -rf distripute/grpc/distripute_pb2*.py
