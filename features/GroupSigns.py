import os
import glob

from features.ExtractFeatures import extract_features


class SignatoryClass:
    def __init__(self, nr=None, session=None, signature_nr=None, user_signature=None, relative_path=None):
        if user_signature is None:
            self.nr = nr
            self.session = session
            self.signature_nr = [signature_nr]
        else:
            self.nr = user_signature[0]
            self.session = user_signature[1]
            self.signature_nr = [user_signature[2]]

        self.relative_path = relative_path

    def signature_path(self):
        return self.relative_path + self.nr

    def compare(self, signatory_nr, relative_path):
        return self.nr == signatory_nr[0] and self.session == signatory_nr[1] and self.relative_path == relative_path

    def append_list(self, new_sing_nr):
        self.signature_nr.append(new_sing_nr)

    def __str__(self):
        return self.nr + '_' + self.session + '_' + str(self.signature_nr)


def group_signature_files(files_urls):
    signatories_container = []

    def known_signatory(signatory_nr, relative_path):
        for signatory in signatories_container:
            if signatory.compare(signatory_nr=signatory_nr, relative_path=relative_path):
                signatory.append_list(signatory_nr[2])
                return
        signatories_container.append(SignatoryClass(user_signature=signatory_nr, relative_path=relative_path))

    for file_url in files_urls:
        basename = os.path.basename(file_url)
        relative_path = file_url.replace(basename, '')
        splited_name = basename.split('_')
        known_signatory(splited_name, relative_path)

    return signatories_container


def get_files(file_name, in_dir_path, out_dir_path, recursively=False):
    if file_name == 'all':

        files_urls = []

        if recursively:
            extension = "/**/*.sig"
        else:
            extension = "/*.sig"

        for file_url in glob.glob(in_dir_path + extension, recursive=recursively):
            files_urls.append(file_url.replace(in_dir_path, ''))
            # files.append(os.path.basename(file))

        users_signatures = group_signature_files(files_urls)

        # print(users_signatures)
        for user_signature in users_signatures:
            files_urls = []
            for signature in user_signature.signature_nr:
                files_urls.append(user_signature.signature_path() + '_' + user_signature.session + '_' + signature)
            extract_features(
                files_urls=files_urls,
                signatory_nr=user_signature.nr,
                in_dir_path=in_dir_path,
                out_dir_path=out_dir_path + user_signature.relative_path,
                session=user_signature.session
            )
