#!/usr/bin/perl -w

# store_dt_on_disk.pl

#use lib '../blib/lib', '../blib/arch';

use strict;
use Algorithm::DecisionTree;
use Storable;

my $training_datafile = "training.dat";


my $dt = Algorithm::DecisionTree->new( 
                              training_datafile => $training_datafile,
                              entropy_threshold => 0.1,
                              max_depth_desired => 3,
#                             debug1 => 1,                          
#                             debug2 => 1,
         );

$dt->get_training_data();


### UNCOMMENT THE NEXT STATEMENT if you would like to see
### the training data that was read from the disk file:
#$dt->show_training_data();


my $root_node = $dt->construct_decision_tree_classifier();

eval {
    store( $dt, 'dt.db' );
};
if ($@) {
    print "Something went wrong with disk storage of decision tree: $@";
}

eval {
    store( $root_node, 'root_node.db' );
};
if ($@) {
    print "Something went wrong with disk storage of root node: $@";
}

