#!/usr/bin/perl -w

#use lib '../blib/lib', '../blib/arch';

use strict;
use Algorithm::DecisionTree;

my $parameter_file = "param_symbolic.txt";
my $output_test_datafile = "testdata_new.dat";
my $output_class_labels_file = "test_data_class_labels_new.dat";



my $test_data_gen = TestDataGeneratorSymbolic->new( 
                     output_test_datafile     => $output_test_datafile,
                     output_class_labels_file => $output_class_labels_file,
                     parameter_file           => $parameter_file,
                     write_to_file            => 1,
                     number_of_test_samples   => 30,
                    );

$test_data_gen->read_parameter_file_symbolic();
$test_data_gen->gen_test_data();
$test_data_gen->write_test_data_to_file(); 
                                           
