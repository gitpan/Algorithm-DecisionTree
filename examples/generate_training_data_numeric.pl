#!/usr/bin/perl -w

## generate_training_data_numeric.pl

#use lib '../blib/lib', '../blib/arch';

use strict;
use Algorithm::DecisionTree;

my $parameter_file = "param_numeric.txt";

my $output_csv_file = "training_new.csv";

my $training_data_gen = TrainingDataGeneratorNumeric->new( 
                              output_csv_file   => $output_csv_file,
                              parameter_file    => $parameter_file,
                              number_of_samples_per_class => 50,
                        );

$training_data_gen->read_parameter_file_numeric();
$training_data_gen->gen_numeric_training_data_and_write_to_csv();



