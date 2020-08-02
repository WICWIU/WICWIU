#include <string>
#include <iostream>
#include <dirent.h>
#include <sys/types.h>

DIR* getDirList(char * dirName)
{
    DIR *dir;
    try
    {
        if ((dir = opendir(dirName)) == NULL)
            throw "could not open directory";
    }
    catch (const char* msg)
    {
        std::cerr << msg << std::endl;
        return nullptr;
    }
    return dir;
}


int main(int argc, char const *argv[])
{
    DIR* dir = getDirList("/tmp/casia_train");
    struct dirent *ent;

    while ((ent = readdir(dir)) != NULL)
    {
        std::cout << ent->d_name << std::endl;
    }
    closedir(dir);
    return EXIT_SUCCESS;
}