import os
import glob

from features.ExtractFeatures import extract_features


class SignatoryClass:
    def __init__(self, nr=None, session=None, signature_nr=None, user_signature=None):
        if user_signature is None:
            self.nr = nr
            self.session = session
            self.signature_nr = [signature_nr]
        else:
            self.nr = user_signature[0]
            self.session = user_signature[1]
            self.signature_nr = [user_signature[2]]

    def compare(self, signatory_nr):
        return self.nr == signatory_nr[0] and self.session == signatory_nr[1]

    def append_list(self, new_sing_nr):
        self.signature_nr.append(new_sing_nr)

    def __str__(self):
        return self.nr + self.session + str(self.signature_nr)


def group_signature_files(files_names):
    signatories_container = []

    def known_signatory(signatory_nr):
        for signatory in signatories_container:
            if signatory.compare(signatory_nr):
                signatory.append_list(signatory_nr[2])
                return
        signatories_container.append(SignatoryClass(user_signature=signatory_nr))

    for file_name in files_names:
        splited_name = file_name.split('_')
        known_signatory(splited_name)

    return signatories_container


def get_files(file_name, in_dir_path, out_dir_path):
    if file_name == 'all':

        files = []

        for file in glob.glob(in_dir_path + "/*.sig"):
            files.append(os.path.basename(file))

        users_signatures = group_signature_files(files)

        # print(users_signatures)
        for user_signature in users_signatures:
            files_urls = []
            for signature in user_signature.signature_nr:
                files_urls.append(in_dir_path + '/' + user_signature.nr + '_' + user_signature.session + '_' + signature)
            extract_features(files_urls=files_urls, signatory_nr=user_signature.nr, out_dir_path=out_dir_path)
