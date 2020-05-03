# Publishing A Jupyter Notebook On Github Pages

In short:

1. Download the notebook as an html file
2. Make a repo on github with a specific name and push the html file to that
   repo

## Walkthrough

1. Get an html file to publish

    1. Restart the kernel and run the notebook from top to bottom
    2. From the jupyter notebook dowload as an html file

2. Make a repo on github that will be your internet site

    [Github Pages][1] is a free service that github offers to host your
    websites.  To get started:
    
    1. [Create a repo on github](https://github.com/new) named exactly
       `yourusername.github.io` (replace `yourusername` with your github
       username).
    2. Create a repository locally and wire it up to the one you created on
       github.
    
    Anything you push to this repository will be live on the internet at
    
      https://yourusername.github.io

3. Add content and make the site live

    1. Copy over the html file you created from the notebook into the new
       repository.
    2. Name the html file index.html (whichever file is named index.html is the
       "landing page" for your site)
    3. Add, commit, and push the index.html file
    
You should now be able to see your site live!, go to
https://yourusername.github.io/ to see it! (it might take 1-5 minutes after
pushing for the site to update)

[1]: https://pages.github.com/

