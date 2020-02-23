SRC_DIR=./protos
DST_DIR=./proto/
protoc -I=$SRC_DIR --python_out=$DST_DIR $SRC_DIR/round.proto