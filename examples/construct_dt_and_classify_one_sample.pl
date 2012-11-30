#!/usr/bin/perl -w

# construct_dt_and_classify_one_sample.pl

#use lib '../blib/lib', '../blib/arch';

use strict;
use Algorithm::DecisionTree;

my $training_datafile = "training.dat";

#   COMMENT OUT the two debug options below if you do not
#   want to see each node of the decision being constructed:
my $dt = Algorithm::DecisionTree->new( 
                              training_datafile => $training_datafile,
                              entropy_threshold => 0.1,
                              max_depth_desired => 3,
#                             debug1            => 1,
#                             debug2            => 1,
         );

$dt->get_training_data();


#   UNCOMMENT THE NEXT STATEMENT if you would like to see the
#   training data that was read from the disk file:
#$dt->show_training_data();


print "\nStarting construction of the decision tree:\n\n";
my $root_node = $dt->construct_decision_tree_classifier();


#   UNCOMMENT THE NEXT STATEMENT if you would like to see the
#   decision tree displayed in your terminal window:
#$root_node->display_decision_tree("     ");           

### The following test_sample is for the training files with names
### like training.dat training2.dat:
my @test_sample = qw /exercising=>never 
                      smoking=>heavy 
                      fatIntake=>heavy 
                      videoAddiction=>heavy /;

#   The classifiy() in the call below returns a reference to a hash
#   whose keys are the class labels and the values the associated 
#   probabilities:
my $classification = $dt->classify($root_node, @test_sample);

#   You can display the classification result by using the
#   following code fragment.  This result will be printed out
#   automatically if you call the constructor new() with
#   debug1 set to 1.
print "\nThe classification:\n";                                        
foreach my $class ($dt->get_class_names()) {
    print "    $class with probability $classification->{$class}\n";    
}                               


