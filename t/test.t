use Test::Simple tests => 2;

use lib '../blib/lib','../blib/arch';

use Algorithm::DecisionTree;

# Test 1 (Digest Parameter File and Generate Training Data):

my $parameter_file = "t/param.txt";
my $output_training_datafile = "t/__trainingdata.dat";
my $output_test_datafile = "t/__testdata.dat";

my $data_gen = TrainingAndTestDataGeneratorSymbolic->new( 
                              output_training_datafile => $output_training_datafile,
                              output_test_datafile  =>  $output_test_datafile,
                              parameter_file    => $parameter_file,
                              number_of_samples_for_training => 35,
                              number_of_samples_for_testing => 5,

               );
$data_gen->read_parameter_file_symbolic();
$data_gen->gen_symbolic_training_and_test_data();
open IN, $output_training_datafile;
my @data_records = <IN>;
ok( @data_records == 51,  'Training data created successfully' );

# Test 2 (Decision Tree Induction):

my $dt = Algorithm::DecisionTree->new(training_datafile => $output_training_datafile);
$dt->get_training_data();
$dt->calculate_first_order_probabilities();
$dt->calculate_class_priors();
my $root_node = $dt->construct_decision_tree_classifier(); 
ok( defined($root_node),  'decision tree induction works' );

unlink "t/__trainingdata.dat";
unlink "t/__testdata.dat";
