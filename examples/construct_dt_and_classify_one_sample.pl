#!/usr/bin/perl -w

#use lib '../blib/lib', '../blib/arch';

use strict;
use Algorithm::DecisionTree;

my $training_datafile = "training.dat";

#   COMMENT OUT the debug option below if you do not want to see
#   each node of the decision being constructed:
my $dt = Algorithm::DecisionTree->new( 
                              training_datafile => $training_datafile,
                              debug1 => 1,                          
#                              entropy_threshold => 0.1
    );

$dt->get_training_data();


#   UNCOMMENT THE FOLLOWING LINE if you would like to see the training
#   data that was read from the disk file:
#$dt->show_training_data();

print "\nStarting construction of the decision tree:\n\n";
my $root_node = $dt->construct_decision_tree_classifier();


#   UNCOMMENT THE FOLLOWING LINE if you would like to see the decision
#   tree displayed in your terminal window:
#$root_node->display_decision_tree("     ");           

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


#   UNCOMMENT THE FOLLOWING TWO LINES if you want to see the number
#   of nodes created:
#my $number_of_nodes = $dt->number_of_nodes_created();
#print "\n\nTotal number of nodes opened:  $number_of_nodes\n";





__END__

# THE COMMENTED OUT CODE THAT IS SHOWN BELOW IS USEFUL FOR DEBUGGING THE
# PROBABILITY AND THE ENTROPY CALCULATION ROUTINES:

#my @features_and_values = qw/ exercising=>never smoking=>heavy fatIntake=>low videoAddiction=>none /;        

#my @features_and_values = qw/ exercising=>regularly smoking=>never fatIntake=>low videoAddiction=>none /;     

#my @features_and_values = qw/ exercising=>regularly smoking=>never fatIntake=>low videoAddiction=>heavy /;     
#print "test prob for sequence @features_and_values\n";



=foralso
my $entro2 = $dt->class_entropy_for_a_given_feature("exercising");
print "\nClass entropy on feature exercising: $entro2\n";

my $entro3 = $dt->class_entropy_for_a_given_feature("videoAddiction");
print "\nClass entropy on feature videoAddiction: $entro3\n";

my $entro5 = $dt->class_entropy_for_a_given_feature("fatIntake");
print "\nClass entropy on feature fatIntake: $entro5\n";
=cut


=foralsoalso
my $entro4 = $dt->class_entropy_for_a_given_feature("smoking");
print "\nClass entropy on feature smoking: $entro4\n";


my $entro6 = $dt->class_entropy_for_a_given_feature_and_given_value("smoking", "never");
print "\nClass entropy with feature smoking with value never: $entro6\n";

my $entro7 = $dt->class_entropy_for_a_given_feature_and_given_value("smoking", "medium");
print "\nClass entropy with feature smoking with value medium: $entro7\n";

my $entro8 = $dt->class_entropy_for_a_given_feature_and_given_value("smoking", "light");
print "\nClass entropy with feature smoking with value light: $entro8\n";

my $entro9 = $dt->class_entropy_for_a_given_feature_and_given_value("smoking", "heavy");
print "\nClass entropy with feature smoking with value heavy: $entro9\n";

my $prob10 = $dt->probability_for_feature_value("smoking", "never");
print "\nProbability for smoking=>never: $prob10\n";

my $prob11 = $dt->probability_for_feature_value("smoking", "medium");
print "\nProbability for smoking=>medium: $prob11\n";

my $prob12 = $dt->probability_for_feature_value("smoking", "light");
print "\nProbability for smoking=>light: $prob12\n";

my $prob13 = $dt->probability_for_feature_value("smoking", "heavy");
print "\nProbability for smoking=>heavy: $prob13\n";
=cut

#my $entro6 = $dt->class_entropy_for_a_given_sequence_of_features_values(@features_and_values);
#print "\nEntropy for the sequence of features and values @features_and_values: $entro6\n";

##my @features_and_values = qw/ exercising=>regularly smoking=>heavy fatIntake=>low videoAddiction=>none /; 

#my $prob3 = $dt->probability_for_a_class_given_feature_value("benign","exercising","never");
#print "\nProbability of class benign on feature exercising and value never: $prob3\n";

#my $prob4 = $dt->probability_for_a_class_given_feature_value("malignant","exercising","never");
#print "\nProbability of class malignant on feature exercising and value never: $prob4\n";

#my $prob6 = $dt->probability_for_a_class_given_sequence_of_features_and_values("malignant", @features_and_values);
#print "\nClass probability of 'malignant' for a sequence of features and values @features_and_values: $prob6\n";

#my $prob7 = $dt->probability_for_a_class_given_sequence_of_features_and_values("benign", @features_and_values);
#print "Class probability of 'benign' for a sequence of features and values @features_and_values: $prob7\n";

#my $prob13 = $dt->probability_for_sequence_of_features_and_values_given_class( "malignant", @features_and_values);
#print "\nProbability for the sequence of features and values @features_and_values on class malignant: $prob13\n";

#my $prob14 = $dt->probability_for_sequence_of_features_and_values_given_class( "benign", @features_and_values);
#print "\nProbability for the sequence of features and values @features_and_values on class benign: $prob14\n";

#my $prob8 = $dt->probability_of_a_sequence_of_features_and_values(@features_and_values);
#print "\nProbability for the sequence of features and values: $prob8\n";


#my $prob10 = $dt->probability_for_feature_value("smoking", "heavy");
#print "Probability for smoking=>heavy: $prob10\n";

#my $prob11 = $dt->probability_for_feature_value("fatIntake", "low");
#print "Probability for fatIntake=>low: $prob11\n";

#my $prob12 = $dt->probability_for_feature_value("videoAddiction", "none");
#print "Probability for videoAddiction=>none: $prob12\n";


