# Lyft Level 5 Prediction Dataset Tools
Provides a set of tools for working with the Prediction Dataset produced by Lyft Level 5 (https://level-5.global/data/prediction/). Note this is largely separate from the python package released by Lyft Level 5 to support the dataset and infact this library is a prerequisite for certain parts of this toolkit to work.

## Prerequisites
The code in this repository was designed to work with Python 3, we offer no guarantees on whether the code will run properly on Python 2 even following conversion by a tool such as 3to2.

In order to use this code, please install l5kit via pip:

pip3 install l5kit

Any other python packages should either be installed by default or included as l5kit prerequisites.

## Convert to Two-Agent Followed / Follower Scene
Converts intermediary agent data stored in an LZ4 compressed custom JSON format to a two-agent convoy scenario scene that causal discovery can be carried out upon. The only difference between the scripts is whether the ego vehicle - which is specified in the supplied agent data - is the lead or tail vehicle in the convoy.

    usage: convert_to_two_agent_followed_scene.py [-h] [--all-kinematic-variables] [--interagent-distance-variables] [--independent-agent-ids [INDEPENDENT_AGENT_IDS [INDEPENDENT_AGENT_IDS ...]]] scene_file_path output_file_path follower_agent_id
           convert_to_two_agent_follower_scene.py [-h] [--all-kinematic-variables] [--interagent-distance-variables] [--independent-agent-ids [INDEPENDENT_AGENT_IDS [INDEPENDENT_AGENT_IDS ...]]] scene_file_path output_file_path followed_agent_id
           
Parameters:
* scene_file_path: File path specifying the input intermediary scene agent data file.
* output_file_path: File path specifying the location to output the resulting scene.
* follower_agent_id: Agent ID of the agent following the ego vehicle.
* followed_agent_id: Agent ID of the agent the ego vehicle is following.
* -h: Displays the help message for the script.
* --all-kinematic-variables: Includes distance travelled and velocity for all scenario agents as variables in the output scene. By default only includes acceleration for all scenario agents.
* --interagent-distance-variables: Includes distance between scenario agents as variables in the output scene.
* --independent-agent-ids: Agent IDs of agents that should be added to the scenario as independent agents with no causal relations.

## Extract Semi-Synthetic Two-Agent Convoy Scenes
Combines intermediary agent data from multiple scenes and extracts two-agent convoy scenario scenes from the resulting data. In order to achieve longer scenes, consecutive scenes are joined together provided a number of safety checks are met. This however introduces the difficulty of finding agents that are both present for the entire scene and devoid of any causal interactions with the convoy agents. This is practically impossible and so this takes a semisynthetic approach by superimposing the data taken from the trajectory of one ego vehicle onto the convoy scene. Thus the independent agent data is sufficiently long and is truly independent, however the resulting scene describes circumstances that never occurred in the real world, despite being based upon real world data.

    usage: extract_semisynthetic_two_agent_convoy_scenes.py [-h] input_dir_path output_dir_path
    
Parameters:
* input_directory_path: Specifies path to directory containing intermediary scene agent data files to take as input.
* output_directory_path: Specifies path to directory to output two-agent convoy scenes to.
* -h: Displays the help message for the script.

## Extract Agents
Takes a scene zarr archive from the base Lyft dataset and outputs intermediary agent data stored in an LZ4 compressed custom JSON format.

    usage: extract_agents.py [-h] scene_dataset_dir_path output_directory_path
    
Parameters:
* input_directory_path: Specifies path to scene zarr archive directory from the base Lyft dataset.
* output_directory_path: Specifies path to directory to output intermediary scene agent data files to.
* -h: Displays the help message for the script.

## Extract Map
Takes protobuf semantic map file, a JSON metadata file and a scene zarr archive and outputs intermediary map data stored in an LZ4 compressed custom JSON format.

    usage: extract_map.py [-h] scene_dataset_dir_path semantic_map_file_path metadata_file_path output_file_path
    
Parameters:
* scene_dataset_dir_path: Specifies path to a scene zarr archive directory in the base Lyft dataset.
* semantic_map_file_path: File path specifying the location of the input protobuf semantic map file.
* metadata_file_path: File path specifying the location of the input JSON metadata file.
* output_file_path: File path specifying the location to output the resulting intermediary map data file.
* -h: Displays the help message for the script.

## Visualise Scene
Takes a scene number and frame number and outputs at least one visualisation image of the scene.

    usage: visualise_scene.py [-h] [--frame-number-offsets [FRAME_NUMBER_OFFSETS [FRAME_NUMBER_OFFSETS ...]]] dataset_directory_path config_file_path scene_number frame_number output_file_path
    
Parameters:
* dataset_directory_path: Specifies path to base Lyft dataset directory.
* config_file_path: Files path specifying the location of the configuration file for the visualisation. The configuration file in turn specifies the location of several other important files/directories in addition to other parameters. An example YAML visualisation configuration file is provided.
* scene_number: Scene ID of scene to visualise.
* frame_number: Base frame ID of frame/s to visualise.
* output_file_path: Base file path specifies the location/s to output the visualisation image/s to.
* -h: Displays the help message for the script.
* --frame-number-offsets: Specifies a list of offsets from the base frame number that visualisation images should be produced for. By default only the base frame is visualised and a single output file is produced at the file path specified. However, if this parameter is specified each visualisation produced will be output to the location described by the base file path suffixed with a label specifying the offset of the frame visualised.

## Other Files
Currently the other files of this toolkit are still in the process of being cleaned up and documented, however these mostly consist of helper classes and functions for the scripts documented here. If more details are required on these files please contact a repository maintainer.
