#!/usr/bin/perl -w

## classify_test_data_in_a_file.pl

#use lib '../blib/lib', '../blib/arch';

use strict;
use Algorithm::DecisionTree;

die "This script must be called with exactly three command-line arguments:\n" .
    "     1st arg: name of the training datafile\n" .
    "     2nd arg: name of the test data file\n" .     
    "     3rd arg: the name of the output file to which class labels will be written\n" 
    unless @ARGV == 3;

my $debug = 0;

my ($training_datafile, $test_datafile, $outputfile) = @ARGV;

my $dt = Algorithm::DecisionTree->new(training_datafile => $training_datafile);

$dt->get_training_data();
$dt->calculate_first_order_probabilities();
$dt->calculate_class_priors();

### UNCOMMENT THE NEXT STATEMENT if you would like to see
### the training data that was read from the disk file:
#$dt->show_training_data();

my $root_node = $dt->construct_decision_tree_classifier();


### UNCOMMENT THE NEXT STATEMENT if you would like to see
### the decision tree displayed in your terminal window:
#$root_node->display_decision_tree("   ");

# NOW YOU ARE READY TO CLASSIFY TEST DATA IN A FILE:

open TESTFILEHANDLE, $test_datafile 
    or die "Unable to open the test datafile $test_datafile: $!";
open OUTPUTHANDLE, ">$outputfile"
    or die "Unable to open the test datafile $outputfile: $!";

my @features;
while (<TESTFILEHANDLE>) {
    chomp;
    next if /^#/;
    next if /^\s*$/;
    if (/^\s*Feature Order For Data:\s*(.+)/i) {
        @features = split /\s+/, $1;
        die "feature labels are empty" if @features == 0;
        next;
    }
    my @test_sample_entries = split /\s+/;
    my $sample_name = shift @test_sample_entries;
    die "the number of features listed in the header does not match " . 
        "the number of values in the test data for sample $sample_name" 
                  unless @features == @test_sample_entries;
    my @test_sample;
    foreach my $i (0..@features-1) {
        push @test_sample, "$features[$i]=$test_sample_entries[$i]";
    }
    my %classification = %{$dt->classify($root_node, \@test_sample)};
    my $result = "$sample_name:  ";
    foreach my $class ($dt->get_class_names()) {
        $result .= sprintf( "$class probability: %.3f    ", 
                            $classification{$class});    
    }
#    print "$result\n" if $debug == 1;
    print "$result\n";
    print OUTPUTHANDLE "$result\n";
}
close TESTFILEHANDLE;
close OUTPUTHANDLE;

