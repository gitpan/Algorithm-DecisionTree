#!/usr/bin/perl -w

#use lib '../blib/lib', '../blib/arch';

use strict;
use Algorithm::DecisionTree;

my $parameter_file = "param3.txt";
my $output_test_datafile = "testdata2.dat";
my $output_class_label_file = "test_data_class_labels.dat";

my $test_data_gen = Algorithm::DecisionTree->test_data_generator( 
                   output_test_datafile    => $output_test_datafile,
                   output_class_label_file => $output_class_label_file,
                   parameter_file          => $parameter_file,
                   write_to_file           => 1,
                   number_of_test_samples  => 20,
    );

$test_data_gen->read_parameter_file();
$test_data_gen->gen_test_data();
$test_data_gen->write_test_data_to_file(); 
                                           

