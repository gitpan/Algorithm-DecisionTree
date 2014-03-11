#!/usr/bin/perl -w

#use lib '../blib/lib', '../blib/arch';

##  generate_training_and_test_data_symbolic.pl

use strict;
use Algorithm::DecisionTree;

my $parameter_file = "param_symbolic.txt";
my $output_training_datafile = "training4.dat";
my $output_test_datafile = "test4.dat";

my $training_data_gen = TrainingAndTestDataGeneratorSymbolic->new( 
                              output_training_datafile => $output_training_datafile,
                              output_test_datafile =>  $output_test_datafile,
                              parameter_file    => $parameter_file,
                              number_of_samples_for_training => 200,
                              number_of_samples_for_testing => 20,
#                              debug => 1
                        );

$training_data_gen->read_parameter_file_symbolic();
$training_data_gen->gen_symbolic_training_and_test_data();

#
