function run_integration () {
        mkdir -p test_keras;
        curl "$1/image_test.py" -L -o test_keras/image_test.py;
        curl "$1/sequence_test.py" -L -o test_keras/sequence_test.py;
        curl "$1/text_test.py" -L -o test_keras/text_test.py;
        py.test test_keras;
}