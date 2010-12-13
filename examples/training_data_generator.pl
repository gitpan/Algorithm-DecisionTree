#!/usr/bin/perl -w

#use lib '../blib/lib', '../blib/arch';

use strict;
use Algorithm::DecisionTree;

my $parameter_file = "param.txt";
my $output_data_file = "training.dat";

my $data_gen = Algorithm::DecisionTree->training_data_generator( 
                              output_datafile => $output_data_file,
                              parameter_file    => $parameter_file,
                              number_of_training_samples => 35,
    );

$data_gen->read_parameter_file();
$data_gen->gen_training_data();
$data_gen->write_training_data_to_file();   # will also display data on 
                                            # terminal if debug1 is on

