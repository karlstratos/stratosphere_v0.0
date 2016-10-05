// Author: Karl Stratos (jlee2071@bloomberg.net)

#include <bilstmtagger.h>

namespace {

using namespace std;

}

int main (int argc, char* argv[]) {
    string model_directory;
    string data_path;
    bool train = false;
    string development_path;
    size_t development_interval = 10000;
    string embedding_path;
    string gazetteer_path;
    size_t dim_char_embeddings = 25;
    size_t dim_word_embeddings = 100;
    size_t dim_gazetteer_embeddings = 1;
    size_t dim_lstm_outputs = 100;
    size_t dim_hidden_layers = 100;
    size_t num_lstm_layers = 1;
    size_t num_epochs = 30;
    string activation = "relu";
    double learning_rate = 0.0004;  // Also try 0.0005, 0.0006, etc.
    double dropout_rate = 0.4;  // Also try 0.2, 0.3, 0.4, 0.5, 0.6
    string performance_measure = "acc";
    string loss_function = "local";
    size_t beam_size = 8;
    bool use_second_lstms = false;
    bool use_gazemb = false;
    bool do_gaztrain = false;
    string prediction_path;
    size_t random_seed = 1278596736;

    // If appropriate, display default options and then close the program.
    bool display_options_and_quit = false;
    for (int i = 1; i < argc; ++i) {
	string arg = (string) argv[i];
	if (arg == "--help" || arg == "-h"){ display_options_and_quit = true; }
    }
    if (argc == 1 || display_options_and_quit) {
	cout << "--model [-]:        \t"
	     << "path to model directory" << endl;
	cout << "--data [-]:        \t"
	     << "path to a data file" << endl;
	cout << "--train:          \t"
	     << "train a model?" << endl;
	cout << "--dev [-]:        \t"
	     << "path to a development data file" << endl;
        cout << "--check [" << development_interval << "]:     \t"
             << "interval to check development accuracy" << endl;
	cout << "--emb:              \t"
	     << "path to pre-trained word embeddings" << endl;
	cout << "--gaz:              \t"
	     << "path to a gazetteer" << endl;
	cout << "--dim-char [" << dim_char_embeddings << "]:   \t"
	     << "dimension of chararacter embeddings" << endl;
	cout << "--dim-word [" << dim_word_embeddings << "]:  \t"
	     << "dimension of word embeddings" << endl;
	cout << "--dim-gaz [" << dim_gazetteer_embeddings << "]:  \t"
	     << "dimension of gazetteer label embeddings" << endl;
	cout << "--dim-lstm [" << dim_lstm_outputs << "]:   \t"
	     << "dimension of LSTM outputs" << endl;
	cout << "--dim-hidden [" << dim_hidden_layers << "]:   \t"
	     << "dimension of hidden layers in feed-foward" << endl;
	cout << "--layer-lstm [" << num_lstm_layers << "]:   \t"
	     << "number of LSTM layers" << endl;
	cout << "--activate [" << activation << "]:    \t"
	     << "activation function" << endl;
	cout << "--lrate [" << learning_rate << "]:    \t"
	     << "learning rate" << endl;
	cout << "--drate [" << dropout_rate << "]:    \t"
	     << "dropout rate" << endl;
	cout << "--epochs [" << num_epochs << "]:     \t"
	     << "number of epochs" << endl;
	cout << "--pred [-]:        \t"
	     << "path to the prediction file" << endl;
	cout << "--perf [" << performance_measure << "]:    \t"
	     << "performance measure" << endl;
	cout << "--loss [" << loss_function << "]:    \t"
	     << "loss function" << endl;
	cout << "--beam [" << beam_size << "]:     \t"
	     << "beam size" << endl;
	cout << "--lstm2:          \t"
	     << "use second LSTMs (word-level)?" << endl;
	cout << "--gazemb:          \t"
	     << "use gazetteer embeddings?" << endl;
	cout << "--gaztrain:          \t"
	     << "do gazetteer regularized training?" << endl;
	cout << "--rand [" << random_seed << "]:  \t"
	     << "random seed (put 0 to randomize this)" << endl;
	cout << "--help, -h:           \t"
	     << "show options and quit?" << endl;
	exit(0);
    }

    // Parse command line arguments.
    for (int i = 1; i < argc; ++i) {
	string arg = (string) argv[i];
	if (arg == "--model") {
	    model_directory = argv[++i];
	} else if (arg == "--data") {
	    data_path = argv[++i];
	} else if (arg == "--train") {
	    train = true;
	} else if (arg == "--dev") {
	   development_path = argv[++i];
	} else if (arg == "--check") {
	   development_interval = stol(argv[++i]);
	} else if (arg == "--emb") {
            embedding_path = argv[++i];
	} else if (arg == "--gaz") {
            gazetteer_path = argv[++i];
	} else if (arg == "--dim-char") {
	    dim_char_embeddings = stol(argv[++i]);
	} else if (arg == "--dim-word") {
	    dim_word_embeddings = stol(argv[++i]);
	} else if (arg == "--dim-gaz") {
	    dim_gazetteer_embeddings = stol(argv[++i]);
	} else if (arg == "--dim-lstm") {
	    dim_lstm_outputs = stol(argv[++i]);
	} else if (arg == "--dim-hidden") {
	    dim_hidden_layers = stol(argv[++i]);
	} else if (arg == "--layer-lstm") {
	    num_lstm_layers = stol(argv[++i]);
	} else if (arg == "--activate") {
	    activation = argv[++i];
	} else if (arg == "--lrate") {
	    learning_rate = stod(argv[++i]);
	} else if (arg == "--drate") {
	    dropout_rate = stod(argv[++i]);
	} else if (arg == "--epochs") {
	    num_epochs = stol(argv[++i]);
	} else if (arg == "--perf") {
	    performance_measure = argv[++i];
	} else if (arg == "--loss") {
	    loss_function = argv[++i];
	} else if (arg == "--beam") {
	    beam_size = stol(argv[++i]);
	} else if (arg == "--lstm2") {
	    use_second_lstms = true;
	} else if (arg == "--gazemb") {
	    use_gazemb = true;
	} else if (arg == "--gaztrain") {
	    do_gaztrain = true;
	} else if (arg == "--pred") {
	   prediction_path = argv[++i];
	} else if (arg == "--rand") {
	    random_seed = stol(argv[++i]);
	} else if (arg == "--cnn-mem") {
            ++i;
	} else {
	    cerr << "Invalid argument \"" << arg << "\": run the command with "
		 << "-h or --help to see possible arguments." << endl;
	    exit(-1);
	}
    }

    // CNN must be initialized before usage...
    if (random_seed > 0) {
        cnn::Initialize(argc, argv, random_seed);
    } else {
        cnn::Initialize(argc, argv);
    }

    BiLSTMTagger tagger(model_directory);
    tagger.set_development_path(development_path);
    tagger.set_development_interval(development_interval);
    tagger.set_embedding_path(embedding_path);
    tagger.set_gazetteer_path(gazetteer_path);
    tagger.set_dim_char_embeddings(dim_char_embeddings);
    tagger.set_dim_word_embeddings(dim_word_embeddings);
    tagger.set_dim_gazetteer_embeddings(dim_gazetteer_embeddings);
    tagger.set_dim_lstm_outputs(dim_lstm_outputs);
    tagger.set_dim_hidden_layers(dim_hidden_layers);
    tagger.set_num_lstm_layers(num_lstm_layers);
    tagger.set_activation(activation);
    tagger.set_learning_rate(learning_rate);
    tagger.set_dropout_rate(dropout_rate);
    tagger.set_num_epochs(num_epochs);
    tagger.set_performance_measure(performance_measure);
    tagger.set_loss_function(loss_function);
    tagger.set_beam_size(beam_size);
    tagger.set_use_second_lstms(use_second_lstms);
    tagger.set_use_gazemb(use_gazemb);
    tagger.set_do_gaztrain(do_gaztrain);

    if (train) {
        tagger.ResetModelDirectory();
        if (do_gaztrain) {
            tagger.TrainGazetteer(data_path);
        } else {
            tagger.Train(data_path);
        }
    }

    tagger.LoadModel();
    double performance = tagger.EvaluatePerformance((train) ?
                                                    development_path :
                                                    data_path,
                                                    prediction_path);
    cerr << performance << endl;
}
