---
layout: page
title:  Brain Extraction with ANTs
date: 2016-06-06 12:00:00
---

Today we will be talking about how to do a brain extraction on T1 anatomical images. Good brain extraction allows better registration of an anatomical image to a standard template and thus better alignment of the functional data to the standard space. Looking for a good brain extraction tool, I’ve asked on Dr. Jeanette Mumford’s Facebook group, and Dr. Chris Gorgolewski recommended ANTs brain extraction. In this post, I will tell you how to set up and use it.

# Installing ANTs

Although ANTs is shipped in [binaries](https://github.com/stnava/ANTs/releases), they did not work on my Ubuntu installation, so I had to use sources, which is the recommended way to install ANTs. Here is how to build and install them [on Linux / Mac OS](https://github.com/stnava/ANTs/wiki/Compiling-ANTs-on-Linux-and-Mac-OS) or [Windows 10](https://github.com/stnava/ANTs/wiki/Compiling-ANTs-on-Windows-10). The building process takes a while.

Do not forget to put ANTs in your environment, specifically put these line to your bash profile:
{% highlight bash %}
export PATH=<path_to_your_home>/antsbin/bin:$PATH
export ANTSPATH=<path_to_your_home>/antsbin/bin
{% endhighlight %}

# Brain Extraction with ANTs

First, you will need to have the antsBrainExtraction.sh script to run the brain extraction. You may find it in the <git directory/Scripts> you cloned on the installation step. If you used the binary files, it probably won’t be there. Therefore, you will need to create an empty file called antsBrainExtraction.sh. Now go [here](https://github.com/stnava/ANTs/blob/master/Scripts/antsBrainExtraction.sh), copy all the code to your file and save it. In any case, you should end up with the shell script for the brain extraction.

Second, you will need to have a template to perform the brain extraction. Dr. Gorgolewski recommended OASIS. Go ahead and download it from [here](https://figshare.com/articles/ANTs_ANTsR_Brain_Templates/915436). Other templates will work too, but it’s up to you to find out which one works the best with your data.

Now we are ready to do the brain extraction itself. Actually, it’s quite simple. All you need is to run the following from your bash terminal:  
{% highlight bash %}
antsBrainExtraction.sh -d <image dimension> -a <anatomical image> \
-e <brainWithSkullTemplate> -m <brainPrior> -o <output>
{% endhighlight %}

Since we use OASIS, here’s what we should have as an actual command:  
{% highlight bash %}
antsBrainExtraction.sh -d 3 -a t1.nii.gz -e T_template0.nii.gz \
-m T_template0_BrainCerebellumProbabilityMask.nii.gz -o output
{% endhighlight %}

Let’s see what’s going on here. `-d` is the dimension of your image, if you have a 3-d image, the value will be 3. ANTs brain extraction does not support 4-d images (yet?), so you cannot run it on timeseries. `-a` is your anatomical T1 image with the brain. `-e` is your OASIS template with the skull. `-m` is the OASIS brain probability mask. For the `-o` you may put anything because this is just a prefix for your output directory and file.

To see other options, run `antsBrainExtraction.sh`. It will give you a nice help.

This will do the brain extraction. Allow it some time. On my home machine, it took ~ 1.5 hours. In the end, you will have the extracted brain and the brain mask. Make sure to inspect them to see if anything weird is going on!

If you have issues with the extraction such as some parts of the eyes or neck left, try using `-f <brainRegistrationMask>` option. In the case of OASIS template, it is `-f T_template0_BrainCerebellumRegistrationMask.nii.gz`. The mask resolved my issues with brain extraction and reduced computation time by half.


