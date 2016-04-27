import requests


def get_cv2_img_array(url_or_path_to_image_file_or_cv2_image_array, convert_url_to_local_filename=False, download=False,
                      download_directory='images', filename=False, replace_https_with_http=True):
    """
    Get a cv2 img array from a number of different possible inputs.

    :param url_or_path_to_image_file_or_cv2_image_array:
    :param convert_url_to_local_filename:
    :param download:
    :param download_directory:
    :return: img_array
    """
    # print('get:' + str(url_or_path_to_image_file_or_cv2_image_array) + ' try local' + str(
    # convert_url_to_local_filename) + ' download:' + str(download))

    # otherwise it's probably a string, check what kind
    if isinstance(url_or_path_to_image_file_or_cv2_image_array, basestring):
        # try getting url locally by changing url to standard name
        if convert_url_to_local_filename:  # turn url into local filename and try getting it again
            # filename = url_or_path_to_image_file_or_cv2_image_array.split('/')[-1].split('#')[0].split('?')[0]
            # jeremy changed this since it didn't work with url -
            # https://encrypted-tbn1.gstatic.com/images?q=tbn:ANd9GcR2oSMcnwErH1eqf4k8fvn2bAxvSdDSbp6voC7ijYJStL2NfX6v
            # TODO: find a better way to create legal filename from url
            filename = \
                url_or_path_to_image_file_or_cv2_image_array.split('/')[-1].split('#')[0].split('?')[-1].split(':')[
                    -1]
            filename = os.path.join(download_directory, filename)
            if filename.endswith('jpg') or filename.endswith('jpeg') or filename.endswith('.bmp') or \
                    filename.endswith('tiff'):
                pass
            else:  # there's no 'normal' filename ending so add .jpg
                filename = filename + '.jpg'
            # print('trying again locally using filename:' + str(filename))
            img_array = get_cv2_img_array(filename, convert_url_to_local_filename=False, download=download,
                                          download_directory=download_directory)
            # maybe return(get_cv2 etc) instead of img_array =
            if img_array is not None:
                # print('got ok array calling self locally')
                return img_array
            else:  # couldnt get locally so try remotely
                # print('trying again remotely since using local filename didnt work, download=' + str( download) + ' fname:' + str(filename))
                return (
                    get_cv2_img_array(url_or_path_to_image_file_or_cv2_image_array, convert_url_to_local_filename=False,
                                      download=download,
                                      download_directory=download_directory))  # this used to be 'return'
        # put images in local directory
        else:
            # get remotely if its a url, get locally if not
            if "://" in url_or_path_to_image_file_or_cv2_image_array:
                if replace_https_with_http:
                    url_or_path_to_image_file_or_cv2_image_array = url_or_path_to_image_file_or_cv2_image_array.replace(
                        "https", "http")
                img_url = url_or_path_to_image_file_or_cv2_image_array
                try:
                    # print("trying remotely (url) ")
                    response = requests.get(img_url)  # download
                    img_array = imdecode(np.asarray(bytearray(response.content)), 1)
                except ConnectionError:
                    logging.warning("connection error - check url or connection")
                    return None
                except:
                    logging.warning(" error other than connection error - check something other than connection")
                    return None

            else:  # get locally, since its not a url
                # print("trying locally (not url)")
                img_path = url_or_path_to_image_file_or_cv2_image_array
                try:
                    img_array = cv2.imread(img_path)
                    if img_array is not None:
                        # print("success trying locally (not url)")
                        got_locally = True
                    else:
                        # print('couldnt get locally (in not url branch)')
                        return None
                except:
                    # print("could not read locally, returning None")
                    logging.warning("could not read locally, returning None")
                    return None  # input isn't a basestring nor a np.ndarray....so what is it?
    else:
        logging.warning("input is neither an ndarray nor a string, so I don't know what to do")
        return None

    # After we're done with all the above, this should be true - final check that we're outputting a good array
    if not (isinstance(img_array, np.ndarray) and isinstance(img_array[0][0], np.ndarray)):
        print("Bad image - check url/path/array:" + str(
            url_or_path_to_image_file_or_cv2_image_array) + 'try locally' + str(
            convert_url_to_local_filename) + ' dl:' + str(
            download) + ' dir:' + str(download_directory))
        logging.warning("Bad image - check url/path/array:" + str(
            url_or_path_to_image_file_or_cv2_image_array) + 'try locally' + str(
            convert_url_to_local_filename) + ' dl:' + str(
            download) + ' dir:' + str(download_directory))
        return (None)
    # if we got good image and need to save locally :
    if download:
        if not got_locally:  # only download if we didn't get file locally
            if not os.path.isdir(download_directory):
                os.makedirs(download_directory)
            if "://" in url_or_path_to_image_file_or_cv2_image_array:  # its a url, get the bifnocho
                if replace_https_with_http:
                    url_or_path_to_image_file_or_cv2_image_array = url_or_path_to_image_file_or_cv2_image_array.replace(
                        "https", "http")
                filename = \
                    url_or_path_to_image_file_or_cv2_image_array.split('/')[-1].split('#')[0].split('?')[-1].split(':')[
                        -1]
                filename = os.path.join(download_directory, filename)
            else:  # its not a url so use straight
                filename = os.path.join(download_directory, url_or_path_to_image_file_or_cv2_image_array)
            if filename.endswith('jpg') or filename.endswith('jpeg') or filename.endswith('.bmp') or filename.endswith(
                    'tiff'):
                pass
            else:  # there's no 'normal' filename ending
                filename = filename + '.jpg'
            try:  # write file then open it
                # print('filename for local write:' + str(filename))
                write_status = imwrite(filename, img_array)
                max_i = 50  # wait until file is readable before continuing
                gotfile = False
                for i in xrange(max_i):
                    try:
                        with open(filename, 'rb') as _:
                            gotfile = True
                    except IOError:
                        time.sleep(10)
                if gotfile == False:
                    print('Could not access {} after {} attempts'.format(filename, str(max_i)))
                    raise IOError('Could not access {} after {} attempts'.format(filename, str(max_i)))
            except:  # this is prob unneeded given the 'else' above
                print('unexpected error in Utils calling imwrite')
    return img_array


