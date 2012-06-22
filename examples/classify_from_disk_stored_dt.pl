#!/usr/bin/perl -w

#use lib '../blib/lib', '../blib/arch';

use strict;
use Algorithm::DecisionTree;
use Storable;

my $dt = retrieve( 'dt.db' );

print "reference type of dt: " . ref($dt) . "\n";

my $root_node = retrieve( 'root_node.db' );

#    UNCOMMENT THE FOLLOWING LINE if you would like to see the decision
#    tree displayed in your terminal window:
#$root_node->display_decision_tree("     ");         

my @test_sample = qw /exercising=>never 
                      smoking=>heavy 
                      fatIntake=>heavy 
                      videoAddiction=>heavy /;


#    The classifiy() in the call below returns a reference to a hash
#    whose keys are the class labels and the values the associated 
#    probabilities:
my $classification = $dt->classify($root_node, @test_sample);

#    You can display the classification result by using the
#    following code fragment.  This result will be printed out
#    automatically if you call the constructor new() with
#    debug1 set to 1.
print "\nThe classification:\n";                                        
foreach my $class ($dt->get_class_names()) {
    print "    $class with probability $classification->{$class}\n";    
}                               


#    Uncomment the following two lines if you want to see the number
#    of nodes created:
#my $number_of_nodes = $dt->number_of_nodes_created();
#print "\n\nTotal number of nodes opened:  $number_of_nodes\n";

