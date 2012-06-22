#!/usr/bin/perl -w

## classify_by_asking_questions.pl

#use lib '../blib/lib', '../blib/arch';

use strict;
use Algorithm::DecisionTree;

my $training_datafile = "training.dat";

#   COMMENT OUT the debug option below if you do not want to see
#   each node of the decision being constructed:
my $dt = Algorithm::DecisionTree->new( 
                              training_datafile => $training_datafile,
#                              entropy_threshold => 0.1,
#                              max_depth_desired => 3,
#                              debug1 => 1,                          
    );

$dt->get_training_data();


#   UNCOMMENT THE NEXT LEFT-JUSTIFIED LINE if you would like to see the
#   training data that was read from the disk file:
#$dt->show_training_data();

#print "\nStarting construction of the decision tree:\n\n";
my $root_node = $dt->construct_decision_tree_classifier();

#   UNCOMMENT THE NEXT LEFT-JUSTIFIED LINE if you would like to see the
#   decision tree displayed in your terminal window:
#$root_node->display_decision_tree("     ");           

#   The classifiy() in the call below returns a reference to a hash
#   whose keys are the class labels and the values the associated 
#   probabilities:
my $classification = $dt->classify_by_asking_questions($root_node);

#   You can display the classification result by using the
#   following code fragment.  This result will be printed out
#   automatically if you call the constructor new() with
#   debug1 set to 1.
print "\nThe classification:\n";                                        
foreach my $class (sort {$classification->{$b} <=> $classification->{$a} } ($dt->get_class_names()) ) {
    print "    $class with probability $classification->{$class}\n";    
}                               

