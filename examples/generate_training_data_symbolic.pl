#!/usr/bin/perl -w

#use lib '../blib/lib', '../blib/arch';

use strict;
use Algorithm::DecisionTree;

my $parameter_file = "param_symbolic.txt";
my $output_data_file = "training_new.dat";



my $training_data_gen = TrainingDataGeneratorSymbolic->new( 
                              output_datafile   => $output_data_file,
                              parameter_file    => $parameter_file,
                              write_to_file     => 1,
                              number_of_training_samples => 50,
                        );

$training_data_gen->read_parameter_file_symbolic();
$training_data_gen->gen_symbolic_training_data();
$training_data_gen->write_training_data_to_file(); 

