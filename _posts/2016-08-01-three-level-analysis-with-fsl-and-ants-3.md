---
layout: page
title:  Three-level analysis with FSL and ANTs in Nipype. Part 3.
date: 2016-08-01 11:05:00
---
Today we will be setting up level 3 in a 3-level model with FSL and Nipype, using data from [Part 1](https://dpaniukov.github.io/2016/07/03/three-level-analysis-with-fsl-and-ants-1.html) and copes from [Part 2](https://dpaniukov.github.io/2016/07/14/three-level-analysis-with-fsl-and-ants-2.html)

Level 3 setup is very similar to levels 1 and 2. At the beginning, we will be importing some useful libraries and setting up the project variables.

{% highlight python %}

#!/usr/bin/env python

import os, sys                                  # system functions
import nipype.interfaces.io as nio           # Data i/o
from nipype.interfaces.io import DataSink
import nipype.interfaces.fsl as fsl          # fsl
import nipype.pipeline.engine as pe          # pypeline engine
import nipype.interfaces.utility as util     # utility
import nipype.algorithms.modelgen as model   # model generation
import errno

fsl.FSLCommand.set_default_output_type('NIFTI_GZ')

"""
Project info
"""
# In case you have only one cope at level 2, specify it on the input anyway.
#Cope number from level 1
lvl1_cope=int(sys.argv[1])
#Cope number from level 2
lvl2_cope=int(sys.argv[2])


project_dir="/mnt/net/LaCie/Analysis/RuleSwitch/"
work_dir="/mnt/net/LaCie/scratch/RuleSwitch/"

model_id='_1'
task_id=1
template_brain = fsl.Info.standard_image('MNI152_T1_2mm_brain.nii.gz')

# The directory where we can find the level 2 output
l12_out_dir=os.path.join(project_dir,"level2s","model"+model_id)

wf = pe.Workflow(name='wf')
wf.base_dir = os.path.join(work_dir,"wdir"+str(model_id)+"lvl3","copes_"+str(lvl1_cope)+"_"+str(lvl2_cope))
wf.config = {"execution": {"crashdump_dir":os.path.join(project_dir,'crashdumps')}}

{% endhighlight %}

Below we will be getting copes, varcopes and masks for subjects. Note there will be just two subjects, but in the real project, there will be many more. Also, recall that in the previous parts of the tutorial we had two tasks. The level 2 contrasts were: task1>baseline, task2>baseline, and task1>task2. Hence the copes were saved appropriately.

{% highlight python %}
subj_copes=[]
subj_varcopes=[]
if lvl2_cope==1: #task1
    subj_copes=[os.path.join(l12_out_dir,"Subject001","stats_dir","_subject_id_Subject001","_flameo"+str(lvl1_cope-1),"stats","cope1.nii.gz"),
                os.path.join(l12_out_dir,"Subject002","stats_dir","_subject_id_Subject002","_flameo"+str(lvl1_cope-1),"stats","cope2.nii.gz")]

    subj_varcopes=[os.path.join(l12_out_dir,"Subject001","stats_dir","_subject_id_Subject001","_flameo"+str(lvl1_cope-1),"stats","varcope1.nii.gz"),
                os.path.join(l12_out_dir,"Subject002","stats_dir","_subject_id_Subject002","_flameo"+str(lvl1_cope-1),"stats","varcope2.nii.gz")]


elif lvl2_cope==2: #task2
    subj_copes=[os.path.join(l12_out_dir,"Subject001","stats_dir","_subject_id_Subject001","_flameo"+str(lvl1_cope-1),"stats","cope2.nii.gz"),
                os.path.join(l12_out_dir,"Subject002","stats_dir","_subject_id_Subject002","_flameo"+str(lvl1_cope-1),"stats","cope1.nii.gz")]

    subj_varcopes=[os.path.join(l12_out_dir,"Subject001","stats_dir","_subject_id_Subject001","_flameo"+str(lvl1_cope-1),"stats","varcope2.nii.gz"),
                os.path.join(l12_out_dir,"Subject002","stats_dir","_subject_id_Subject002","_flameo"+str(lvl1_cope-1),"stats","varcope1.nii.gz")]

elif lvl2_cope==3: #task1>task2
    subj_copes=[os.path.join(l12_out_dir,"Subject001","stats_dir","_subject_id_Subject001","_flameo"+str(lvl1_cope-1),"stats","cope3.nii.gz"),
                os.path.join(l12_out_dir,"Subject002","stats_dir","_subject_id_Subject002","_flameo"+str(lvl1_cope-1),"stats","cope4.nii.gz")]

    subj_varcopes=[os.path.join(l12_out_dir,"Subject001","stats_dir","_subject_id_Subject001","_flameo"+str(lvl1_cope-1),"stats","varcope3.nii.gz"),
                os.path.join(l12_out_dir,"Subject002","stats_dir","_subject_id_Subject002","_flameo"+str(lvl1_cope-1),"stats","varcope4.nii.gz")]

subj_masks=[os.path.join(l12_out_dir,"Subject001","stats_dir","_subject_id_Subject001","_flameo"+str(lvl1_cope-1),"stats","mask.nii.gz"),
            os.path.join(l12_out_dir,"Subject002","stats_dir","_subject_id_Subject002","_flameo"+str(lvl1_cope-1),"stats","mask.nii.gz")]

{% endhighlight %}

Getting subject info. Do not forget that we have to have the condition_key_l3.txt file and task_contrasts_l3.txt files with condition and task info.  

{% highlight python %}
def get_subjectinfo(subj_list, base_dir, task_id, model_id):
    #from glob import glob
    import os
    import numpy as np

    evs_l3=dict(ev001=[1]*len(subj_list))

    #Conditions for level 3
    condition_info = []
    cond_file = os.path.join(base_dir, 'models', 'model%s' % model_id,
                             'condition_key_l3.txt')
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

    return task_id, evs_l3, conds[task_id - 1]

subjinfo = pe.Node(util.Function(input_names=['subj_list','base_dir', 'task_id', 'model_id'],
                                output_names=['task_id','evs_l3','conds'],
                                function=get_subjectinfo),
                       name='subjectinfo')
subjinfo.inputs.base_dir = project_dir
subjinfo.inputs.task_id = task_id
subjinfo.inputs.model_id = model_id
subjinfo.inputs.subj_list = subj_copes

{% endhighlight %}

Getting data

{% highlight python %}
datasource = pe.Node(nio.DataGrabber(infields=['model_id'], outfields=['contrasts_l3']), name='grabber')
datasource.inputs.base_directory = project_dir
datasource.inputs.template = '*'
datasource.inputs.field_template = dict(contrasts_l3='models/model%s/task_contrasts_l3.txt')
datasource.inputs.template_args = dict(contrasts_l3=[['model_id']])
datasource.inputs.sort_filelist=True
datasource.inputs.model_id=model_id

{% endhighlight %}

Like in level 2, merging the copes, varcopes and masks

{% highlight python %}
copemerge    = pe.Node(interface=fsl.Merge(dimension='t'),
                          name="copemerge")

varcopemerge = pe.Node(interface=fsl.Merge(dimension='t'),
                       name="varcopemerge")

maskemerge = pe.Node(interface=fsl.Merge(dimension='t'),
                       name="maskemerge")

copemerge.inputs.in_files=subj_copes
varcopemerge.inputs.in_files=subj_varcopes
maskemerge.inputs.in_files=subj_masks

{% endhighlight %}

Setup a set of contrasts for level 3

{% highlight python %}

def get_contrasts_l3(contrast_file, task_id, conds):
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

contrastgen_l3 = pe.Node(util.Function(input_names=['contrast_file',
                                                'task_id', 'conds'],
                                   output_names=['contrasts'],
                                   function=get_contrasts_l3),
                      name='contrastgen_l3')

wf.connect(subjinfo, 'conds', contrastgen_l3, 'conds')
wf.connect(datasource, 'contrasts_l3', contrastgen_l3, 'contrast_file')
wf.connect(subjinfo, 'task_id', contrastgen_l3, 'task_id')

{% endhighlight %}


Use MultipleRegressDesign to generate subject and condition
specific level 3 model design files

{% highlight python %}

level3model = pe.Node(interface=fsl.MultipleRegressDesign(),
                      name='l3model')

wf.connect(contrastgen_l3, 'contrasts', level3model, 'contrasts')
wf.connect(subjinfo, 'evs_l3', level3model, 'regressors')

{% endhighlight %}

Saving

{% highlight python %}
datasink = pe.Node(nio.DataSink(), name='sinker')
datasink.inputs.base_directory=os.path.join(project_dir, "level3s","model"+model_id,"copes_"+str(lvl1_cope)+"_"+str(lvl2_cope))

wf.connect(copemerge, 'merged_file', datasink, 'copes_merged')
wf.connect(level3model, 'design_con', datasink, 'design_con')
wf.connect(level3model, 'design_mat', datasink, 'design_mat')

{% endhighlight %}

Running

{% highlight python %}

results = wf.run()

{% endhighlight %}

After it's done running, we need to use FSL's `randomise` to get pretty brain pictures. In `randomise` we will be using standard MNI brain mask from FSL, which copied to the project directory/masks/WB.nii.gz, although you can use the mask from the level 3 above (and my goal was to show you how to make it in case you do not want to use the MNI mask). Below there are two `randomise` commands for two copes. This is just an example because we should be randomizing all the copes we created.

{% highlight bash %}
#!/bin/bash
set proj_dir="/mnt/net/LaCie/Analysis/RuleSwitch/"

mkdir ${proj_dir}level3s/model_1/randomized

randomise -i ${proj_dir}level3s/model_1/copes_1_1/copes_merged/cope1_merged.nii.gz -m ${proj_dir}masks/WB.nii.gz -d ${proj_dir}level3s/model_1/copes_1_1/design_mat/design.mat -t ${proj_dir}level3s/model_1/copes_1_1/design_con/design.con -o ${proj_dir}level3s/model_1/randomized/rand_1_1 -v 8 -C 2.49 &

randomise -i ${proj_dir}level3s/model_1/copes_1_2/copes_merged/cope2_merged.nii.gz -m ${proj_dir}masks/WB.nii.gz -d ${proj_dir}level3s/model_1/copes_1_2/design_mat/design.mat -t ${proj_dir}level3s/model_1/copes_1_2/design_con/design.con -o ${proj_dir}level3s/model_1/randomized/rand_1_2 -v 8 -C 2.49 &

{% endhighlight %}

I would like to thank Dr. Tyler Davis for his guidance and help with FSL and quality checking. Of course, all possible inaccuracies in the code are mine.

