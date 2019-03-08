import xml.etree.ElementTree as ET

location_dict = {}  #made global so that it can be used in the later method mapLocationNameWithImageId
def mapLocationIdWithName(dirPath):
    tree = ET.parse(dirPath+'/devset_topics.xml')
    root = tree.getroot()
    for topic in root.iter('topic'):
        number = topic.find('number').text
        title = topic.find('title').text
        location_dict[number] = title
    return location_dict

def mapLocationNameWithImageId(dirPath):
    location_dict = mapLocationIdWithName(dirPath);
    locationImageMap = {}
    for k, v in location_dict.items():
        tree = ET.parse(dirPath +'/xml/'  +v+'.xml')
        root = tree.getroot()
        for topic in root.iter('photo'):
            imageId = topic.get('id')
            locationImageMap[imageId] = v
    return locationImageMap
