---
layout: page
title:  Three-level analysis with FSL and ANTs in Nipype. Part 1.
date: 2016-07-03 13:10:00
---
In a series of posts I plan to talk about how to run three-level analysis with FSL and ANTs. We will use [ANTs](https://github.com/stnava/ANTs) for registration, [FSL](www.fmrib.ox.ac.uk/fsl) for the analysis itself and [nipype](http://nipy.org/nipype/index.html) for putting everything together. I will be heavily utilizing a code from [nipype examples](http://nipy.org/nipype/documentation.html), changing it's when necessary. Again, this is not an original work, it is rather putting everything together and modifying when it's appropriate.

To illustrate the analysis, I will use a study (manuscript in preparation) on category learning. It had two tasks in counterbalanced order (task 1 and task 2). Each task was scanned within four runs. In the code, I will put comments that explain what is going on and why I apply exceptions to some of my subjects or to the nipype original code.

## Registration with ANTs  

At this post we will be doing registration with ANTs. The overall idea is to run the registration once and then run whatever number of analysis you want, applying this registration.

### Preparation and preprocessing

First, let's import some interfaces and libraries we will be using below.

{% highlight python %}
# -*- coding: utf-8 -*-
#!/usr/bin/env python
import os,sys
import nipype.interfaces.io as nio
import nipype.interfaces.fsl as fsl
import nipype.pipeline.engine as pe
import nipype.interfaces.utility as util
import nipype.interfaces.ants as ants
from nipype.interfaces.c3 import C3dAffineTool
{% endhighlight %}

Assign output type for FSL, so it will create files `.nii.gz`:
{% highlight python %}
fsl.FSLCommand.set_default_output_type('NIFTI_GZ')
{% endhighlight %}

Now, let's define some variables specific to our project. First, we will be needing the root project directory, where all our data are.

{% highlight python %}
project_dir="<path_to_directory>"
{% endhighlight %}

Then, we will also need a working directory, where we can put all temporary files, created by nipype. This directory should be specific for each analysis (alternatively, you can change a name of a workflow for each analysis), and may be deleted as soon as the analysis is finished. Besides, if your analyses use the same working directory and the node names overlap, most probably they will not run properly.
{% highlight python %}
work_dir="<path_to_directory>"
if not os.path.exists(work_dir):
    os.makedirs(work_dir)
{% endhighlight %}

I prefer to input from the command line which subjects to run, because it is easier to parallel your jobs.
{% highlight python %}
subj_list=str(sys.argv[1])
{% endhighlight %}

Now, let's define the workflow.
{% highlight python %}
wf = pe.Workflow(name='wf')
wf.base_dir = os.path.join(work_dir,"reg_wdir")
wf.config = {"execution": {"crashdump_dir":os.path.join(work_dir,'reg_crashdumps')}}
{% endhighlight %}

... and get a subject's id:
{% highlight python %}
infosource = pe.Node(interface=util.IdentityInterface(fields=['subject_id']), name="infosource")
infosource.iterables = ('subject_id', [subj_list])
{% endhighlight %}
Again, I am using just one subject per running job, so they are already in parallel. But in case you want to put all of them into this script and run in parallel (may the power of CPU be with you!), this is the place to do it (e.g., put `["Subject001","Subject002"]` instead of `[subj_list]`, and remove `subj_list` variable above.)

Here I use a function to get information for a specific subject. It may be different information such as ids for run numbers, condition information, etc. In the code below, some subjects had no run 1 (because of a visual spike, it was removed), and some had no last runs (did not finish the experiment).
{% highlight python %}
def get_subjectinfo(subject_id):

    run_id=[]
    if subject_id=="Subject003" or subject_id=="Subject011" or subject_id=="Subject016" or subject_id=="Subject020":
        run_id=["2","3","4","5","6","7","8"]
    elif subject_id=="Subject019":
        run_id=["1","2","3","4","5","6"]
    else:
        run_id=["1","2","3","4","5","6","7","8"]

    return run_id

subjinfo = pe.Node(util.Function(input_names=['subject_id'],
                                output_names=['run_id'],
                                function=get_subjectinfo),
                       name='subjectinfo')

wf.connect([(infosource, subjinfo, [('subject_id', 'subject_id')]),])
{% endhighlight %}

Now, it is time to get all the files we need from the hard drive.
{% highlight python %}
datasource = pe.Node(nio.DataGrabber(infields=['subject_id','run_id'], outfields=['func', 'anat']), name='datasource')
datasource.inputs.base_directory = project_dir
datasource.inputs.template = '*'
datasource.inputs.field_template = dict(func='%s/bold/run%s/run*_mcf_brain.nii.gz',
                                        anat='%s/anatomy/highres001_BrainExtractionBrain.nii.gz')
datasource.inputs.template_args = dict(func=[['subject_id','run_id']],
                                       anat=[['subject_id']])
datasource.inputs.sort_filelist=True

wf.connect(subjinfo, 'subject_id', datasource, 'subject_id')
wf.connect(subjinfo, 'run_id', datasource, 'run_id')
{% endhighlight %}

Ok, we have grabbed the files and now we can get the middle volume from each run for the functional to anatomical registration.

Define interfaces first:
{% highlight python %}
inputnode = pe.Node(interface=util.IdentityInterface(fields=['func',
                                                             'anat',]),
                    name='inputspec')

wf.connect([(datasource, inputnode, [('anat','anat'),('func', 'func'),]),])
{% endhighlight %}

Convert functional images to float representation. Since there can be more than one functional run we use a MapNode to convert each run.
{% highlight python %}
prefiltered_func_data = pe.MapNode(interface=fsl.ImageMaths(out_data_type='float',
                                             op_string = '',
                                             suffix='_dtype'),
                       iterfield=['in_file'],
                       name='prefiltered_func_data')

wf.connect(inputnode, 'func', prefiltered_func_data, 'in_file')
{% endhighlight %}

Extract the middle volume of the run as the reference and define a function to return the 1 based index of the middle volume.
{% highlight python %}
example_func = pe.MapNode(interface=fsl.ExtractROI(t_size=1),
                          iterfield=['in_file'],
                          name = 'example_func')

def getmiddlevolume(func):
    from nibabel import load
    funcfile = func
    if isinstance(func, list):
        funcfile = func[0]
    _,_,_,timepoints = load(funcfile).get_shape()
    return (timepoints/2)-1

wf.connect(prefiltered_func_data, 'out_file', example_func, 'in_file')
wf.connect(inputnode, ('func', getmiddlevolume), example_func, 't_min')
{% endhighlight %}

### Register functionals to anatomical space

Estimate the tissue classes from the anatomical image.
{% highlight python %}
fast = pe.Node(fsl.FAST(), name='fast')

wf.connect(inputnode, 'anat', fast, 'in_files')
{% endhighlight %}

Binarize the segmentation.
{% highlight python %}
binarize = pe.Node(fsl.ImageMaths(op_string='-nan -thr 0.5 -bin'),
                   name='binarize')
pickindex = lambda x, i: x[i]

wf.connect(fast, ('partial_volume_files', pickindex, 2), binarize, 'in_file')
{% endhighlight %}

Calculate rigid transform from example_func image to anatomical image.
{% highlight python %}
func2anat = pe.MapNode(fsl.FLIRT(), iterfield=['in_file'], name='func2anat')
func2anat.inputs.dof = 6

wf.connect(example_func, 'roi_file', func2anat, 'in_file')
wf.connect(inputnode, 'anat', func2anat, 'reference')
{% endhighlight %}

Now use BBR cost function to improve the transform.
{% highlight python %}
func2anatbbr = pe.MapNode(fsl.FLIRT(), iterfield=['in_file','in_matrix_file'], name='func2anatbbr')
func2anatbbr.inputs.dof = 6
func2anatbbr.inputs.cost = 'bbr'
func2anatbbr.inputs.schedule = os.path.join(os.getenv('FSLDIR'),'etc/flirtsch/bbr.sch')

wf.connect(example_func, 'roi_file', func2anatbbr, 'in_file')
wf.connect(binarize, 'out_file', func2anatbbr, 'wm_seg')
wf.connect(inputnode, 'anat', func2anatbbr, 'reference')
wf.connect(func2anat, 'out_matrix_file', func2anatbbr, 'in_matrix_file')
{% endhighlight %}

Convert the BBRegister transformation to ANTS ITK format for further reuse.
{% highlight python %}
convert2itk = pe.MapNode(C3dAffineTool(), iterfield=['source_file','transform_file'], name='convert2itk')
convert2itk.inputs.fsl2ras = True
convert2itk.inputs.itk_transform = True

wf.connect(func2anatbbr, 'out_matrix_file', convert2itk, 'transform_file')
wf.connect(example_func, 'roi_file',convert2itk, 'source_file')
wf.connect(inputnode, 'anat', convert2itk, 'reference_file')
{% endhighlight %}

We are done with the registration of functional files for now. We will use the affine matrix later to transform the example_func to standard space.

### Register anatomical to standard space

You can find a nice crash course on the ANTs registration [here](https://www.neuroinf.jp/fmanager/view/737/bah20150723-alex.pdf). Oh, keep in mind that this example uses 12 CPU threads for a single subject registration. In case you want to run 25 subjects in this script in parallel, you should adjust `reg.inputs.num_threads` to the number your computer can handle.

{% highlight python %}
template_brain = fsl.Info.standard_image('MNI152_T1_2mm_brain.nii.gz')

reg = pe.Node(ants.Registration(), name='antsRegister')
reg.inputs.output_transform_prefix = "output_"
reg.inputs.transforms = ['Rigid', 'Affine', 'SyN']
reg.inputs.transform_parameters = [(0.1,), (0.1,), (0.2, 3.0, 0.0)]
reg.inputs.number_of_iterations = [[100000, 111100, 111100]] * 2 + [[1000, 300, 200]]
reg.inputs.dimension = 3
reg.inputs.write_composite_transform = True
reg.inputs.collapse_output_transforms = False
reg.inputs.initial_moving_transform_com = True
reg.inputs.metric = ['Mattes'] * 2 + [['Mattes', 'CC']]
reg.inputs.metric_weight = [1] * 2 + [[0.5, 0.5]]
reg.inputs.radius_or_number_of_bins = [32] * 2 + [[32, 4]]
reg.inputs.sampling_strategy = ['Regular'] * 2 + [[None, None]]
reg.inputs.sampling_percentage = [0.3] * 2 + [[None, None]]
reg.inputs.convergence_threshold = [1.e-8] * 2 + [-0.01]
reg.inputs.convergence_window_size = [20] * 2 + [5]
reg.inputs.smoothing_sigmas = [[4, 2, 1]] * 2 + [[1, 0.5, 0]]
reg.inputs.sigma_units = ['vox'] * 3
reg.inputs.shrink_factors = [[3, 2, 1]]*2 + [[4, 2, 1]]
reg.inputs.use_estimate_learning_rate_once = [True] * 3
reg.inputs.use_histogram_matching = [False] * 2 + [True]
reg.inputs.winsorize_lower_quantile = 0.005
reg.inputs.winsorize_upper_quantile = 0.995
reg.inputs.args = '--float'
reg.inputs.output_warped_image = 'output_warped_image.nii.gz'
reg.inputs.output_inverse_warped_image = 'output_inverse_warped_image.nii.gz'
reg.inputs.num_threads = 12

reg.inputs.fixed_image=template_brain
wf.connect(inputnode, 'anat', reg, 'moving_image')
{% endhighlight %}

### Warp functionals to standard space

Strictly speaking, we will be warping only middle volume for each run (example_func) to standard space for quality assessment.  
Concatenate the affine and ants transforms into a list.
{% highlight python %}
merge = pe.MapNode(util.Merge(2), iterfield=['in2'], name='mergexfm')

wf.connect(convert2itk, 'itk_transform', merge, 'in2')
wf.connect(reg, 'composite_transform', merge, 'in1')
{% endhighlight %}

Transform the example_func image, first to anatomical and then to target.
{% highlight python %}
warp_func = pe.MapNode(ants.ApplyTransforms(), iterfield=['input_image','transforms'], name='warp_func')
warp_func.inputs.input_image_type = 0
warp_func.inputs.interpolation = 'Linear'
warp_func.inputs.invert_transform_flags = [False, False]
warp_func.inputs.terminal_output = 'file'

wf.connect(example_func, 'roi_file', warp_func, 'input_image')
wf.connect(merge, 'out', warp_func, 'transforms')
warp_func.inputs.reference_image = template_brain
{% endhighlight %}

### Save the data and run

We need to save all our data, don't we?! Here we will save the warped anatomical image from ANTs registration, its inverse, both regular non-linear and inverse non-linear ANTs transform martices, transformed example_func to standard space, functional to anatomical space matrices, functional to standard matrices, and the example_func itself just in case we will need it in the future (for computing betaseries in mvpa analysis, for example).

{% highlight python %}
datasink = pe.Node(nio.DataSink(), name='sinker')
datasink.inputs.base_directory=os.path.join(project_dir, "reg")

wf.connect(subjinfo, 'subject_id', datasink, 'container')
wf.connect(reg, 'warped_image', datasink, 'anatomy.anat2standard')
wf.connect(reg, 'inverse_warped_image', datasink, 'anatomy.standard2anat')
wf.connect(reg, 'composite_transform', datasink, 'anatomy.anat2standard_mat')
wf.connect(reg, 'inverse_composite_transform', datasink, 'anatomy.standard2anat_mat')
wf.connect(warp_func, 'output_image', datasink, 'bold.func2standard')
wf.connect(func2anatbbr, 'out_matrix_file', datasink, 'bold.func2anat_transform')
wf.connect(merge, 'out', datasink, 'bold.func2standard_mat')
wf.connect(example_func, 'roi_file', datasink, 'bold.example_func')
{% endhighlight %}

Shall we run it? For single CPU computer remove `plugin='MultiProc'`, put `reg.inputs.num_threads = 1` and do not try to run subjects in parallel!
{% highlight python %}
results = wf.run(plugin='MultiProc')
{% endhighlight %}

### Quality assessment

For the quality assessment, revisit all warped anatomical images and functional images transformed to standard space. For the functional images you can also use FSL's `slices` in a terminal (in bash, not in python!) as:
{% highlight bash %}
slices your_BOLD_image.nii.gz MNI152_T1_2mm_brain.nii.gz -o output_image.gif
{% endhighlight %}

Please email me your comments and questions!
