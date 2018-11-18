---
layout: page
title:  Three-level analysis with FSL and ANTs in Nipype. Part 2.
date: 2016-07-14 17:28:00
---
Today we will be setting up levels 1 and 2 in a 3-level model with FSL and Nipype, applying transformation matrices from ANTs that we created in [Part 1](https://dpaniukov.github.io/2016/07/03/three-level-analysis-with-fsl-and-ants-1.html).

## Onset and contrast files
Before we start running anything, we have to put some files in places. First, all level 1 onset files should be in `<project directory>/<subject directory>/<model>/<model specific folder>/onsets/<task#_run#>`. Next, we have to have a folder inside your main project directory called `models`. Under this folder, we will create a model specific folder with a description of EVs and contrasts. Thus, let's create a folder `<project directory>/models/model_1`. Now let's create the following files in there.
For level 1 we should have a file called `condition_key.txt` with the following content:
{% highlight text %}
task001 ev001 Stimuli_correct
task001 ev002 Stimuli_incorrect
task001 ev003 Stim_na
task001 ev004 Feedback_correct
task001 ev005 Feedback_incorrect
task001 ev006 Feedback_na
{% endhighlight %}
As you see in our example, we have 6 EVs, and should create contrasts like these in the `task_contrasts.txt`:
{% highlight text %}
task001 stim_cor 1 0 0 0 0 0
task001 stim_incor 0 1 0 0 0 0
task001 stim_cor>stim_inc 1 -1 0 0 0 0
task001 fdb_cor 0 0 0 1 0 0
task001 fdb_incor 0 0 0 0 1 0
task001 fdb_cor>fdb_incor 0 0 0 1 -1 0
{% endhighlight %}
This is how you create EVs and contrasts in Nipype. Let's do it for level 2. By the way, the example experiment had two tasks.
`condition_key_l2.txt`:
{% highlight text %}
task001 ev001 task1
task001 ev002 task2
{% endhighlight %}
... and contrasts in `task_contrasts_l2.txt` to compare one task with another:
{% highlight text %}
task001 task1 1 0
task001 task2 0 1
task001 task1>task2 1 -1
{% endhighlight %}

Ok, all external files are in place, so we can move to our main analysis script.

## Preparation and Preprocessing

We need to import some libraries and interfaces to work with:
{% highlight python %}
# -*- coding: utf-8 -*-
#!/usr/bin/env python
import os,sys
import nipype.interfaces.io as nio
import nipype.interfaces.fsl as fsl
import nipype.interfaces.ants as ants
import nipype.pipeline.engine as pe
import nipype.interfaces.utility as util
import nipype.algorithms.modelgen as model
import errno
{% endhighlight %}

Assign output type for FSL, so it will create files `.nii.gz`:
{% highlight python %}
fsl.FSLCommand.set_default_output_type('NIFTI_GZ')
{% endhighlight %}

Now, let's define some variables specific to our project. First, we will be needing the root project directory, where all our data are.

{% highlight python %}
project_dir="<path_to_directory>"
model_id='_1'
task_id=1
TR=2.0
fwhm_thr=6.0 #smoothing kernel
hpcutoff = 100 # highpass filter cutoff
film_thr=1000 #default in FSL
film_ms=5 # this should be Susan mask size, not fwhm
template_brain = fsl.Info.standard_image('MNI152_T1_2mm_brain.nii.gz')
{% endhighlight %}

Then, we will also need a working directory, where we can put all temporary files, created by nipype. This directory should be specific for each analysis (alternatively, you can change the name of a workflow for each analysis), and may be deleted as soon as the analysis is finished. Besides, if your analyses use the same working directory and the node names overlap, most probably they will not run properly.
{% highlight python %}
work_dir="<path_to_directory>"
try:
    os.mkdir(work_dir)
except OSError as exc:
    if exc.errno != errno.EEXIST or exc.errno != 17:
        raise exc
    pass
{% endhighlight %}

I prefer to input from the command line which subjects to run because it is easier to parallel your jobs.
{% highlight python %}
subj_list=str(sys.argv[1])
{% endhighlight %}

Now, let's define the workflow.
{% highlight python %}
wf = pe.Workflow(name='wf')
wf.base_dir = os.path.join(work_dir,"wdir"+str(model_id)+"lvl12")
wf.config = {"execution": {"crashdump_dir":os.path.join(project_dir,'crashdumps')}}
{% endhighlight %}

... and get a subject's id:
{% highlight python %}
infosource = pe.Node(interface=util.IdentityInterface(fields=['subject_id']), name="infosource")
infosource.iterables = ('subject_id', [subj_list])
{% endhighlight %}
Again, I am using just one subject per running job, so they are already in parallel. But in case you want to put all of them into this script and run in parallel (may the power of CPU be with you!), this is the place to do it (e.g., put `["Subject001","Subject002"]` instead of `[subj_list]`, and remove `subj_list` variable above.)

Here I use a function to get information for a specific subject. It may be different information such as ids for run numbers, condition information, etc.
{% highlight python %}
def get_subjectinfo(subject_id, base_dir, task_id, model_id):
    import os
    import numpy as np

    if subject_id=="Subject003" or subject_id=="Subject011" or subject_id=="Subject016" or subject_id=="Subject020":
        run_id=[2,3,4,5,6,7,8]
        itk_id=list(np.array(run_id)-2)
        evs_l2=dict(ev001=[1,1,1,0,0,0,0], ev002=[0,0,0,1,1,1,1])
    elif subject_id=="Subject019":
        run_id=[1,2,3,4,5,6]
        itk_id=list(np.array(run_id)-1)
        evs_l2=dict(ev001=[1,1,1,1,0,0], ev002=[0,0,0,0,1,1])
    else:
        run_id=[1,2,3,4,5,6,7,8]
        itk_id=list(np.array(run_id)-1)
        evs_l2=dict(ev001=[1,1,1,1,0,0,0,0], ev002=[0,0,0,0,1,1,1,1])

    #Conditions for level 1
    condition_info = []
    cond_file = os.path.join(base_dir, 'models', 'model%s' % model_id,
                             'condition_key.txt')
    with open(cond_file, 'rt') as fp:
        for line in fp:
            info = line.strip().split()
            condition_info.append([info[0], info[1], ' '.join(info[2:])])
    if len(condition_info) == 0:
        raise ValueError('No condition info found in %s' % cond_file)
    taskinfo = np.array(condition_info)
    n_tasks = len(np.unique(taskinfo[:, 0]))
    conds = []
    if task_id > n_tasks:
        raise ValueError('Task id %d does not exist' % task_id)
    for idx in range(n_tasks):
        taskidx = np.where(taskinfo[:, 0] == 'task%03d' % (idx + 1))
        conds.append([condition.replace(' ', '_') for condition
                      in taskinfo[taskidx[0], 2]])

    #Conditions for level 2
    condition_info_l2 = []
    cond_file_l2 = os.path.join(base_dir, 'models', 'model%s' % model_id,
                             'condition_key_l2.txt')
    with open(cond_file_l2, 'rt') as fp_l2:
        for line in fp_l2:
            info_l2 = line.strip().split()
            condition_info_l2.append([info_l2[0], info_l2[1], ' '.join(info_l2[2:])])
    if len(condition_info_l2) == 0:
        raise ValueError('No condition info found in %s' % cond_file_l2)
    taskinfo_l2 = np.array(condition_info_l2)
    n_tasks_l2 = len(np.unique(taskinfo_l2[:, 0]))
    conds_l2 = []
    if task_id > n_tasks_l2:
        raise ValueError('Task id %d does not exist' % task_id)
    for idx in range(n_tasks_l2):
        taskidx_l2 = np.where(taskinfo_l2[:, 0] == 'task%03d' % (idx + 1))
        conds_l2.append([condition_l2.replace(' ', '_') for condition_l2
                      in taskinfo_l2[taskidx_l2[0], 2]])

    return subject_id, model_id, task_id, run_id, conds[task_id - 1], itk_id, evs_l2, conds_l2[task_id - 1],
{% endhighlight %}
Let's stop for a moment and take a look at what's going on here. Some subjects had no run 1 (because of a visual spike, it was removed), and some had no last runs (did not finish the experiment), thus `run_id` is different for some subjects. `itk_id` was made to indicate an appropriate run. Nipype counts the runs starting at 0, while I counted them starting at 1, thus, the run numbers are adjusted for files made by Nipype earlier. `evs_l2` is a vector of ones (1) for each of two tasks. In most of the cases you will be using only one EV at level 2 to combine runs across subjects; and you will have something like this: `evs_l2=dict(ev001=[1,1,1,1,1,1,1,1])`. Everything under that just parses out the condition files for each run and each level.

Prepare the node to run the function above and grab subject-specific data as we did in the [Part 1](https://dpaniukov.github.io/2016/07/03/three-level-analysis-with-fsl-and-ants-1.html):
{% highlight python %}
subjinfo = pe.Node(util.Function(input_names=['subject_id','base_dir', 'task_id', 'model_id'],
                                output_names=['subject_id','model_id','task_id','run_id','conds','itk_id','evs_l2','conds_l2'],
                                function=get_subjectinfo),
                       name='subjectinfo')
subjinfo.inputs.base_dir = project_dir
subjinfo.inputs.task_id = task_id
subjinfo.inputs.model_id = model_id

datasource = pe.Node(nio.DataGrabber(infields=['subject_id','model_id','task_id','run_id','itk_id'],
                                              outfields=['func', 'struct','behave','contrasts',
                                              'contrasts_l2','confound','itk_transform',
                                              'composite_transform']), name='grabber')
datasource.inputs.base_directory = project_dir
datasource.inputs.template = '*'
datasource.inputs.field_template = dict(func='%s/bold/run%d/run*_mcf_brain.nii.gz',
                                        struct='%s/anatomy/highres001_BrainExtractionBrain.nii.gz',
                                        behave='%s/model/model%s/onsets/task%03d_run%d/ev*.txt',
                                        contrasts='models/model%s/task_contrasts.txt',
                                        contrasts_l2='models/model%s/task_contrasts_l2.txt',
                                        confound='%s/bold/run%d/QA/confound3.txt',
                                        itk_transform='reg/%s/bold/func2standard_mat/_subject_id_%s/_convert2itk%d/affine.txt',
                                        composite_transform='reg/%s/anatomy/anat2standard_mat/_subject_id_%s/output_Composite.h5')
datasource.inputs.template_args = dict(func=[['subject_id','run_id']],
                                       struct=[['subject_id']],
                                        behave=[['subject_id','model_id','task_id','run_id']],
                                        contrasts=[['model_id']],
                                        contrasts_l2=[['model_id']],
                                        confound=[['subject_id','run_id']],
                                        itk_transform=[['subject_id','subject_id','itk_id']],
                                        composite_transform=[['subject_id','subject_id']])
datasource.inputs.sort_filelist=True

wf.connect([(infosource, subjinfo, [('subject_id', 'subject_id')]),])
wf.connect(subjinfo, 'subject_id', datasource, 'subject_id')
wf.connect(subjinfo, 'model_id', datasource, 'model_id')
wf.connect(subjinfo, 'task_id', datasource, 'task_id')
wf.connect(subjinfo, 'run_id', datasource, 'run_id')
wf.connect(subjinfo, 'itk_id', datasource, 'itk_id')
{% endhighlight %}
Here we import functional and structural scans, onset files, contrast files, and transformation matrices.

Set up a node to define all inputs required for the preprocessing workflow
{% highlight python %}
inputnode = pe.Node(interface=util.IdentityInterface(fields=['func','struct',]),
                    name='inputspec')

wf.connect([(datasource, inputnode, [('struct','struct'),('func', 'func'),]),])
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

Determine the 2nd and 98th percentile intensities of each functional run

{% highlight python %}
getthresh = pe.MapNode(interface=fsl.ImageStats(op_string='-p 2 -p 98'),
                       iterfield = ['in_file'],
                       name='getthreshold')

wf.connect(prefiltered_func_data, 'out_file', getthresh, 'in_file')
{% endhighlight %}

Threshold the first run of the functional data at 10% of the 98th percentile

{% highlight python %}
threshold = pe.MapNode(interface=fsl.ImageMaths(out_data_type='char',
                                             suffix='_thresh'),
                    iterfield = ['in_file'],
                    name='threshold')
{% endhighlight %}

Define a function to get 10% of the intensity

{% highlight python %}
def getthreshop(thresh):
    return '-thr %.10f -Tmin -bin'%(0.1*thresh[0][1])

wf.connect(prefiltered_func_data, 'out_file', threshold, 'in_file')
wf.connect(getthresh, ('out_stat', getthreshop), threshold, 'op_string')
{% endhighlight %}

Determine the median value of the functional runs using the mask

{% highlight python %}
medianval = pe.MapNode(interface=fsl.ImageStats(op_string='-k %s -p 50'),
                       iterfield = ['in_file','mask_file'],
                       name='medianval')

wf.connect(prefiltered_func_data, 'out_file', medianval, 'in_file')
wf.connect(threshold, 'out_file', medianval, 'mask_file')
{% endhighlight %}

Dilate the mask. This is the final mask for level 1.

{% highlight python %}
dilatemask = pe.MapNode(interface=fsl.ImageMaths(suffix='_dil',
                                              op_string='-dilF'),
                    iterfield=['in_file'],
                     name='dilatemask')

wf.connect(threshold, 'out_file', dilatemask, 'in_file')
{% endhighlight %}

Mask the motion corrected functional runs with the dilated mask
{% highlight python %}
prefiltered_func_data_thresh = pe.MapNode(interface=fsl.ImageMaths(suffix='_mask',
                                                op_string='-mas'),
                       iterfield=['in_file','in_file2'],
                       name='prefiltered_func_data_thresh')

wf.connect(prefiltered_func_data, 'out_file', prefiltered_func_data_thresh, 'in_file')
wf.connect(dilatemask, 'out_file', prefiltered_func_data_thresh, 'in_file2')
{% endhighlight %}

Determine the mean image from each functional run

{% highlight python %}
meanfunc2 = pe.MapNode(interface=fsl.ImageMaths(op_string='-Tmean',
                                                suffix='_mean'),
                       iterfield=['in_file'],
                       name='meanfunc2')

wf.connect(prefiltered_func_data_thresh, 'out_file', meanfunc2, 'in_file')
{% endhighlight %}

Merge the median values with the mean functional images into a coupled list

{% highlight python %}
#Yes, it is Node with iterfield! Not MapNode.
mergenode = pe.Node(interface=util.Merge(2, axis='hstack'),
                       iterfield=['in1','in2'],
                       name='merge')

wf.connect(meanfunc2,'out_file', mergenode, 'in1')
wf.connect(medianval,'out_stat', mergenode, 'in2')
{% endhighlight %}

Smooth each run using SUSAN with the brightness threshold set to 75% of the median value for each run and a mask constituting the mean functional

{% highlight python %}
smooth = pe.MapNode(interface=fsl.SUSAN(),
                    iterfield=['in_file', 'brightness_threshold','usans'],
                    name='smooth')
smooth.inputs.fwhm = fwhm_thr
{% endhighlight %}

Here I have to note that if you will be replicating the same in FSL, you won't get exact same copes. The reason is that Nipype uses a standard formula to calculate smoothing: `sqrt(8*log(2))` when FSL uses a different formula. Thus, to completely replicate FSL, you should use `fwhm_thr=5.999541516002418` instead of `fwhm_thr=6.0` we used above.

Define a function to get the brightness threshold for SUSAN

{% highlight python %}
def getbtthresh(medianvals):
    return [0.75*val for val in medianvals]

def getusans(x):
    return [[tuple([val[0],0.75*val[1]])] for val in x]

wf.connect(prefiltered_func_data_thresh, 'out_file', smooth, 'in_file')
wf.connect(medianval, ('out_stat', getbtthresh), smooth, 'brightness_threshold')
wf.connect(mergenode, ('out', getusans), smooth, 'usans')
{% endhighlight %}

Mask the smoothed data with the dilated mask

{% highlight python %}
maskfunc3 = pe.MapNode(interface=fsl.ImageMaths(suffix='_mask',
                                                op_string='-mas'),
                       iterfield=['in_file', 'in_file2'],
                       name='maskfunc3')

wf.connect(smooth, 'smoothed_file', maskfunc3, 'in_file')
wf.connect(dilatemask, 'out_file', maskfunc3, 'in_file2')
{% endhighlight %}

Scale each volume of the run so that the median value of the run is set to 10000

{% highlight python %}
intnorm = pe.MapNode(interface=fsl.ImageMaths(suffix='_intnorm'),
                     iterfield=['in_file','op_string'],
                     name='intnorm')
{% endhighlight %}

Define a function to get the scaling factor for intensity normalization

{% highlight python %}
def getinormscale(medianvals):
    return ['-mul %.10f'%(10000./val) for val in medianvals]

wf.connect(maskfunc3, 'out_file', intnorm, 'in_file')
wf.connect(medianval, ('out_stat', getinormscale), intnorm, 'op_string')
{% endhighlight %}

Create tempMean

{% highlight python %}
tempMean = pe.MapNode(interface=fsl.ImageMaths(op_string='-Tmean',
                                                suffix='_mean'),
                       iterfield=['in_file'],
                       name='tempMean')

wf.connect(intnorm, 'out_file', tempMean, 'in_file')
{% endhighlight %}

Perform temporal highpass filtering on the data. This is the same as `filtered_func_data` in FSL output.

{% highlight python %}
highpass = pe.MapNode(interface=fsl.ImageMaths(op_string= '-bptf %d -1 -add'%(hpcutoff/(2*TR)), suffix='_tempfilt'),
                      iterfield=['in_file','in_file2'],
                      name='highpass')

wf.connect(tempMean, 'out_file', highpass, 'in_file2')
wf.connect(intnorm, 'out_file', highpass, 'in_file')
{% endhighlight %}

We've done with preprocessing! Hooray! Now, let's see how to code the level 1.

## Level 1

Setup a basic set of contrasts

{% highlight python %}
def get_contrasts(contrast_file, task_id, conds):
    import numpy as np
    contrast_def = np.genfromtxt(contrast_file, dtype=object)
    if len(contrast_def.shape) == 1:
        contrast_def = contrast_def[None, :]
    contrasts = []
    for row in contrast_def:
        if row[0] != 'task%03d' % task_id:
            continue
        con = [row[1], 'T', ['ev%03d' % (i + 1)  for i in range(len(conds))],
               row[2:].astype(float).tolist()]
        contrasts.append(con)
    return contrasts

contrastgen = pe.Node(util.Function(input_names=['contrast_file',
                                                'task_id', 'conds'],
                                   output_names=['contrasts'],
                                   function=get_contrasts),
                      name='contrastgen')

wf.connect(subjinfo, 'conds', contrastgen, 'conds')
wf.connect(datasource, 'contrasts', contrastgen, 'contrast_file')
wf.connect(subjinfo, 'task_id', contrastgen, 'task_id')
{% endhighlight %}

Generate design information

{% highlight python %}
def check_behav_list(behav):
    from nipype.external import six
    out_behav = []
    if isinstance(behav, six.string_types):
        behav = [behav]
    for val in behav:
        if not isinstance(val, list):
            out_behav.append([val])
        else:
            out_behav.append(val)
    return out_behav

modelspec = pe.MapNode(interface=model.SpecifyModel(),
                      iterfield=['event_files','functional_runs'],
                      name="modelspec")
modelspec.inputs.input_units = 'secs'
modelspec.inputs.high_pass_filter_cutoff = hpcutoff
modelspec.inputs.time_repetition=TR

wf.connect(datasource, ('behave', check_behav_list), modelspec, 'event_files')
wf.connect(highpass, 'out_file', modelspec, 'functional_runs')
{% endhighlight %}

Generate a run specific .fsf file for analysis

{% highlight python %}
level1design = pe.MapNode(interface=fsl.Level1Design(),
                          iterfield=['session_info'], name="level1design")
level1design.inputs.interscan_interval=TR
level1design.inputs.bases = {'dgamma':{'derivs': True}}
level1design.inputs.model_serial_correlations = True

wf.connect(contrastgen, 'contrasts', level1design, 'contrasts')
wf.connect(modelspec, 'session_info', level1design, 'session_info')
{% endhighlight %}

Generate a run specific .mat file for use by FILMGLS

{% highlight python %}
modelgen = pe.MapNode(interface=fsl.FEATModel(), name='modelgen',
                      iterfield = ['fsf_file', 'ev_files','args'])

wf.connect(level1design, 'ev_files', modelgen, 'ev_files')
wf.connect(level1design, 'fsf_files', modelgen, 'fsf_file')
wf.connect(datasource, 'confound', modelgen, 'args')
{% endhighlight %}

Estimate a model specified by a .mat file and a functional run

{% highlight python %}
modelestimate = pe.MapNode(interface=fsl.FILMGLS(smooth_autocorr=True,
                                                 mask_size=film_ms,
                                                 threshold=film_thr),
                           name='modelestimate',
                           iterfield = ['design_file','in_file','tcon_file'])

wf.connect([(highpass,modelestimate,[('out_file','in_file')]),
            (modelgen,modelestimate,[('design_file','design_file')]),
   ])
wf.connect(modelgen, 'con_file', modelestimate, 'tcon_file')
{% endhighlight %}

The output from `modelestimate` is what we see in FSL level 1 analysis.

## Level 2

One of the main issues we may see in Nipepe examples is that they recommend first running FLAMEO in level 2, and then registering its outputs to the standard space. Well, if you assume that subjects never move between runs, this might be appropriate. When was the last time you saw subjects being completely still?! Rhetorical question... Thus, we will be applying our registration matrices right after the copes, varcopes, etc. were created in level 1, then merging them and running FLAMEO on them.

{% highlight python %}
merge_mat = pe.MapNode(util.Merge(2), iterfield=['in2'], name='merge_mat')

wf.connect(datasource, 'itk_transform', merge_mat, 'in2')
wf.connect(datasource, 'composite_transform', merge_mat, 'in1')


def warp_files(copes, varcopes, masks, mat, template_brain):
    import nipype.interfaces.ants as ants

    out_copes=[]
    out_varcopes=[]
    out_masks=[]

    warp = ants.ApplyTransforms()
    warp.inputs.input_image_type = 0
    warp.inputs.interpolation = 'Linear'
    warp.inputs.invert_transform_flags = [False,False]
    warp.inputs.terminal_output = 'file'
    warp.inputs.reference_image = template_brain
    warp.inputs.transforms=mat

    warp.inputs.input_image = masks
    res=warp.run()
    out_masks.append(str(res.outputs.output_image))

    for cope in copes:
        warp.inputs.input_image = cope
        res=warp.run()
        out_copes.append(str(res.outputs.output_image))

    for varcope in varcopes:
        warp.inputs.input_image = varcope
        res=warp.run()
        out_varcopes.append(str(res.outputs.output_image))

    return out_copes, out_varcopes, out_masks

warpfunc = pe.MapNode(util.Function(input_names=['copes', 'varcopes', 'masks', 'mat','template_brain'],
                               output_names=['out_copes', 'out_varcopes', 'out_masks'],
                               function=warp_files),
                               iterfield=['copes', 'varcopes', 'masks','mat'],
                  name='warpfunc')

warpfunc.inputs.template_brain = template_brain
wf.connect(modelestimate, 'copes', warpfunc, 'copes')
wf.connect(modelestimate, 'varcopes', warpfunc, 'varcopes')
wf.connect(dilatemask, 'out_file', warpfunc, 'masks')
wf.connect(merge_mat, 'out', warpfunc, 'mat')
{% endhighlight %}
As you see, we do not do anything different from registering example_func in [Part 1](https://dpaniukov.github.io/2016/07/03/three-level-analysis-with-fsl-and-ants-1.html).

Now merge the copes, varcopes and masks for each condition

{% highlight python %}
copemerge    = pe.MapNode(interface=fsl.Merge(dimension='t'),
                          iterfield=['in_files'],
                          name="copemerge")

varcopemerge = pe.MapNode(interface=fsl.Merge(dimension='t'),
                       iterfield=['in_files'],
                       name="varcopemerge")

maskemerge = pe.MapNode(interface=fsl.Merge(dimension='t'),
                       iterfield=['in_files'],
                       name="maskemerge")

def sort_copes(files):
    numelements = len(files[0])
    outfiles = []
    for i in range(numelements):
        outfiles.insert(i,[])
        for j, elements in enumerate(files):
            outfiles[i].append(elements[i])
    return outfiles

wf.connect(warpfunc, ('out_copes',sort_copes), copemerge, 'in_files')
wf.connect(warpfunc, ('out_varcopes',sort_copes), varcopemerge, 'in_files')
wf.connect(warpfunc, ('out_masks',sort_copes), maskemerge, 'in_files')
{% endhighlight %}

Setup a set of contrasts for level 2

{% highlight python %}
def get_contrasts_l2(contrast_file, task_id, conds):
    import numpy as np
    contrast_def = np.genfromtxt(contrast_file, dtype=object)
    if len(contrast_def.shape) == 1:
        contrast_def = contrast_def[None, :]
    contrasts = []
    for row in contrast_def:
        if row[0] != 'task%03d' % task_id:
            continue
        con = [row[1], 'T', ['ev%03d' % (i + 1)  for i in range(len(conds))],
               row[2:].astype(float).tolist()]
        contrasts.append(con)
    return contrasts

contrastgen_l2 = pe.Node(util.Function(input_names=['contrast_file',
                                                'task_id', 'conds'],
                                   output_names=['contrasts'],
                                   function=get_contrasts_l2),
                      name='contrastgen_l2')

wf.connect(subjinfo, 'conds_l2', contrastgen_l2, 'conds')
wf.connect(datasource, 'contrasts_l2', contrastgen_l2, 'contrast_file')
wf.connect(subjinfo, 'task_id', contrastgen_l2, 'task_id')
{% endhighlight %}

Here we will use `MultipleRegressDesign` instead of `L2Model` recommended in Nipype examples to generate subject and condition-specific level 2 model design files because `MultipleRegressDesign` can handle more than one EV according to its name... ((c) Captain Obvious)

{% highlight python %}
level2model = pe.Node(interface=fsl.MultipleRegressDesign(),
                      name='l2model')

wf.connect(contrastgen_l2, 'contrasts', level2model, 'contrasts')
wf.connect(subjinfo, 'evs_l2', level2model, 'regressors')
{% endhighlight %}

Estimate a second level model with FLAMEO

{% highlight python %}
flameo = pe.MapNode(interface=fsl.FLAMEO(run_mode='fe'), name="flameo",
                    iterfield=['cope_file','var_cope_file'])

pickfirst = lambda x: x[0]

wf.connect([(copemerge,flameo,[('merged_file','cope_file')]),
          (varcopemerge,flameo,[('merged_file','var_cope_file')]),
          (level2model,flameo, [('design_mat','design_file'),
                                ('design_con','t_con_file'),
                                ('design_grp','cov_split_file')]),
          (maskemerge,flameo,[(('merged_file',pickfirst),'mask_file')]),
          ])
{% endhighlight %}

Saving output since we are running this code for a single subject and will be combining data across subjects in level 3. Also, let's save filtered_func_data just in case we need it sometime later, like to create means for timeseries and use them as inputs in PPI.

{% highlight python %}
datasink = pe.Node(nio.DataSink(), name='sinker')
datasink.inputs.base_directory=os.path.join(project_dir, "level2s","model"+model_id)

wf.connect(infosource, 'subject_id', datasink, 'container')
wf.connect(highpass, 'out_file', datasink, 'filtered_func_data')
wf.connect([(flameo, datasink, [('stats_dir', 'stats_dir')])])
{% endhighlight %}

Run, Forrest....
{% highlight python %}
results = wf.run(plugin='MultiProc')
{% endhighlight %}

This is it for the preprocessing, levels 1 and 2. Next time, I will post my code for level 3 with some explanations.

I would like to thank Dr. Tyler Davis for his guidance and help with FSL and quality checking. Of course, all possible inaccuracies in the code are mine.


