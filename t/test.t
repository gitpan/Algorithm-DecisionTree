use Test::Simple tests => 2;

use lib '../blib/lib','../blib/arch';

use Algorithm::DecisionTree;

# Test 1 (Digest Parameter File and Generate Training Data):

my $parameter_file = "param.txt";
my $output_data_file = "__testdata.dat";

my $data_gen = Algorithm::DecisionTree->training_data_generator( 
                              output_datafile => $output_data_file,
                              parameter_file    => $parameter_file,
                              number_of_training_samples => 35,
                              debug1 => 0,
                              debug2 => 0,
    );

$data_gen->read_parameter_file();
$data_gen->gen_training_data();
$data_gen->write_training_data_to_file(); 
open IN, $output_data_file;
my @data_records = <IN>;
ok( @data_records == 51,  'Training data created successfully' );

# Test 2 (Decision Tree Induction):

my $dt = Algorithm::DecisionTree->new( 
             training_datafile => $output_data_file,
                                     );
$dt->get_training_data();
my $root_node = $dt->construct_decision_tree_classifier(); 
ok( defined($root_node),  'decision tree induction works' );

unlink "__testdata.dat";
