// Author: Karl Stratos (stratos@cs.columbia.edu)

#include "arguments.h"
#include "singular.h"

int main (int argc, char* argv[]) {
    ArgumentProcessor argparser;
    argparser.ParseArguments(argc, argv);

    /*
    // Initialize a Singular object.
    Singular singular(argparser.output_directory());
    singular.set_rare_cutoff(argparser.rare_cutoff());
    singular.set_sentence_per_line(argparser.sentence_per_line());
    singular.set_window_size(argparser.window_size());
    singular.set_context_definition(argparser.context_definition());
    singular.set_dim(argparser.dim());
    singular.set_transformation_method(argparser.transformation_method());
    singular.set_scaling_method(argparser.scaling_method());
    singular.set_num_context_hashed(argparser.num_context_hashed());
    singular.set_pseudocount(argparser.pseudocount());
    singular.set_context_smoothing_exponent(
	argparser.context_smoothing_exponent());
    singular.set_singular_value_exponent(argparser.singular_value_exponent());
    singular.set_verbose(argparser.verbose());


    // If given a corpus, extract statistics from it.
    if (!argparser.corpus_path().empty()) {
	if (argparser.from_scratch()) { singular.ResetOutputDirectory(); }
	singular.ExtractStatistics(argparser.corpus_path());
    }

    // Induce word representations from cached statistics.
    singular.InduceLexicalRepresentations();
    */
}
