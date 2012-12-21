#!/usr/bin/perl -w

#use lib '../blib/lib', '../blib/arch';

use strict;
use Algorithm::DecisionTree;

my $parameter_file = "param3.txt";
my $output_data_file = "training3.dat";

my $training_data_gen = Algorithm::DecisionTree->training_data_generator( 
                              output_datafile   => $output_data_file,
                              parameter_file    => $parameter_file,
                              write_to_file     => 1,
                              number_of_training_samples => 50,
#                             debug1            => 1,
#                             debug2            => 1,
                        );

$training_data_gen->read_parameter_file();
$training_data_gen->gen_training_data();
$training_data_gen->write_training_data_to_file(); 


